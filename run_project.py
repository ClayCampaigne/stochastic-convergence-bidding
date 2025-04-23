# --- START OF CORRECTED run_project.py with Price Functions Moved ---

import time
import numpy as np
import sys
import os
import concurrent.futures
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable, Optional, NamedTuple, Any # Added Callable, List
from dataclasses import dataclass, field
import csv
import argparse
import xarray as xr # Added xr for price functions
import cvxpy as cp # Added for solver reference

# --- Loguru Setup ---
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Set max workers for parallel processing
MAX_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4


from stochastic_convergence_bidding.bidding_model import BiddingModel, BidPriceFunction # BidPriceFunction still needed if defined there, or define here
from stochastic_convergence_bidding.market_data import MarketData
from stochastic_convergence_bidding.sample_data_generator import generate_sample_data

np.random.seed(1)

# Create output directory if it doesn't exist
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cache for sample data to avoid regenerating in the same process
sample_data_cache: Dict[Tuple[int, int], Tuple[np.ndarray, List[str]]] = {}


def use_all_unique_prices() -> BidPriceFunction:
    """
    Creates a function that returns all unique DA prices from the dataset.

    Returns:
        A function that takes an hour's data and returns unique prices.
    """
    def get_prices(hour_data: xr.Dataset) -> np.ndarray:
        # Ensure 'dalmp' exists and handle potential NaNs
        if "dalmp" not in hour_data:
            logger.error("Variable 'dalmp' not found in hour_data for use_all_unique_prices.")
            return np.array([]) # Return empty array
        prices = hour_data["dalmp"].values
        return np.unique(prices[~np.isnan(prices)]) # Filter NaNs before unique
    return get_prices

def evenly_spaced_sample_based_prices(num_bid_prices: int) -> BidPriceFunction:
    """
    Creates a function that returns evenly spaced prices between min and max observed prices.

    Args:
        num_bid_prices: Number of evenly spaced price points to create.

    Returns:
        A function that takes an hour's data and returns evenly spaced prices.
    """
    if num_bid_prices <= 0:
        raise ValueError("num_bid_prices must be positive for evenly_spaced_sample_based_prices")
    def get_prices(hour_data: xr.Dataset) -> np.ndarray:
        if "dalmp" not in hour_data:
            logger.error("Variable 'dalmp' not found in hour_data for evenly_spaced_sample_based_prices.")
            return np.array([])
        prices = hour_data["dalmp"].values
        valid_prices = prices[~np.isnan(prices)]
        if valid_prices.size == 0:
            logger.warning(f"No valid 'dalmp' prices found for hour {hour_data['hour'].item()} in evenly_spaced_sample_based_prices.")
            return np.array([])
        min_price = np.min(valid_prices)
        max_price = np.max(valid_prices)
        if min_price == max_price: # Handle case where all prices are the same
            return np.array([min_price])
        # Use num_bid_prices for linspace if > 1, otherwise return single price
        return np.linspace(min_price, max_price, max(1, num_bid_prices))
    return get_prices

def fixed_grid_prices(price_grid: List[float]) -> BidPriceFunction:
    """
    Creates a function that returns a fixed grid of prices regardless of the dataset.

    Args:
        price_grid: List of fixed price points to use.

    Returns:
        A function that ignores the hour's data and returns the fixed price grid.
    """
    # Convert to numpy array on creation for efficiency
    np_price_grid = np.array(price_grid, dtype=np.float64)
    if np_price_grid.size == 0:
        logger.warning("fixed_grid_prices created with an empty price grid.")
    def get_prices(hour_data: xr.Dataset) -> np.ndarray:
        # Ignore hour_data, return the stored grid
        return np_price_grid
    return get_prices

# --- Data Structures ---

@dataclass(frozen=True)
class PriceStrategy:
    """Encapsulates a function to get bid prices and its description."""
    function: BidPriceFunction
    description: str
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AnalysisCase:
    """Parameters defining a single analysis run."""
    case_id: str
    n_scenarios: int
    price_strategy: PriceStrategy
    risk_constraint: bool

@dataclass
class CaseResult:
    """Results from a single analysis run."""
    case: AnalysisCase
    revenue: Optional[float] = None
    solve_time_s: Optional[float] = None
    total_time_s: Optional[float] = None
    status: Optional[str] = None
    error_message: Optional[str] = None

# --- Price Strategy Factory ---

def create_price_strategy(
    strategy_type: str,
    num_points: Optional[int] = None,
    fixed_resolution: Optional[float] = None, # Allow float resolution
    fixed_grid_list: Optional[List[float]] = None,
    min_price: float = -100.0,
    max_price: float = 200.0
) -> PriceStrategy:
    """Creates a PriceStrategy object based on configuration."""
    config = locals()
    if strategy_type == "all_unique":
        func = use_all_unique_prices()
        desc = "All Unique"
    elif strategy_type == "evenly_spaced":
        if num_points is None: raise ValueError("num_points missing for 'evenly_spaced'")
        func = evenly_spaced_sample_based_prices(num_points)
        desc = f"{num_points} Evenly Spaced"
    elif strategy_type == "fixed_resolution":
        if fixed_resolution is None: raise ValueError("fixed_resolution missing for 'fixed_resolution'")
        if fixed_resolution <= 0: raise ValueError("fixed_resolution must be positive")
        # Use np.arange for potentially fractional steps, include endpoint carefully
        grid = list(np.arange(min_price, max_price + fixed_resolution/2.0, fixed_resolution)) # Adjust endpoint logic if needed
        func = fixed_grid_prices(grid)
        desc = f"Fixed Grid Res ${fixed_resolution}" # Add $ sign for clarity
        config['generated_grid_list'] = grid
    elif strategy_type == "fixed_list":
        if fixed_grid_list is None: raise ValueError("fixed_grid_list missing for 'fixed_list'")
        func = fixed_grid_prices(fixed_grid_list)
        desc = "Fixed Grid Custom List"
    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")
    return PriceStrategy(function=func, description=desc, config=config)

# --- Analysis Case Generation ---
def create_analysis_cases(
    scenario_list: List[int],
    price_strategy_configs: List[Dict[str, Any]],
    use_risk_constraint: bool
) -> List[AnalysisCase]:
    """Generates a list of AnalysisCase objects to be run."""
    cases = []
    case_counter = 0
    for n_scenarios in scenario_list:
        for strategy_config in price_strategy_configs:
            try:
                price_strategy = create_price_strategy(**strategy_config)

                if price_strategy.config.get('strategy_type') == 'evenly_spaced':
                    num_points = price_strategy.config.get('num_points')
                    if num_points is not None and num_points >= n_scenarios:
                        logger.info(f"Skipping case: {n_scenarios} scenarios, {price_strategy.description} (num_points >= n_scenarios)")
                        continue

                risk_label = 'risk' if use_risk_constraint else 'norisk'
                # Sanitize description for filename
                strat_label = price_strategy.description.replace(' ', '_').replace('$', '').replace('.', 'p')
                case_id = f"case_{case_counter:04d}_{n_scenarios}s_{strat_label}_{risk_label}"
                case = AnalysisCase(
                    case_id=case_id,
                    n_scenarios=n_scenarios,
                    price_strategy=price_strategy,
                    risk_constraint=use_risk_constraint
                )
                cases.append(case)
                case_counter += 1
            except ValueError as e:
                logger.warning(f"Skipping case generation due to error: {e}")
            except Exception as e:
                 logger.exception(f"Unexpected error generating case for {strategy_config}: {e}")

    logger.info(f"Generated {len(cases)} analysis cases.")
    return cases


# --- Core Optimization Runner ---
def run_optimization(
    n_scenarios: int,
    get_bid_prices_f: BidPriceFunction,
    risk_constraint: bool,
    alpha: float = 0.95,
    rho: float = -1000.0,
    hours_list: List[int] = list(range(24)),
    verbose_market_data: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[BiddingModel]]:
    pid = os.getpid()
    try:
        cache_key = (pid, n_scenarios)
        if cache_key in sample_data_cache: data, target_names = sample_data_cache[cache_key]
        else:
            data, target_names = generate_sample_data(num_samples=n_scenarios, num_hours=max(hours_list) + 1, random_seed=0)
            sample_data_cache[cache_key] = (data, target_names)
        market_data = MarketData(data, target_names)
        if verbose_market_data: market_data.print_hourly_report()
    except Exception as e:
        logger.error(f"[{pid}] Error generating/loading data for {n_scenarios} scenarios: {e}")
        return None, None, "data_error", None

    model = None
    solve_time_s: Optional[float] = None
    try:
        model = BiddingModel(market_data=market_data, hours=hours_list, get_bid_prices_f=get_bid_prices_f, alpha=alpha, rho=rho, verbose=False)
        model.build_model(risk_constraint=risk_constraint)
        start_solve_time = time.time()
        model.solve_model(solver=cp.CLARABEL) # Replace cp.CLARABEL if needed
        solve_time_s = time.time() - start_solve_time
        status = "unknown"; objective_value = None
        if model.problem is not None:
             status = model.problem.status; objective_value = model.objective_value
             if status not in ["optimal", "optimal_inaccurate"]:
                 logger.warning(f"[{pid}] Solver status for {n_scenarios} scenarios, {get_bid_prices_f.__name__}: {status}")
                 objective_value = objective_value if objective_value is not None else np.nan
        else: status = "build_failed?"
        return objective_value, solve_time_s, status, model
    except Exception as e:
        logger.exception(f"[{pid}] Error during optimization model execution for {n_scenarios} scenarios: {e}")
        return None, solve_time_s, "model_error", model


# --- Parallel Execution Worker ---
def _run_single_case_worker(case: AnalysisCase) -> CaseResult:
    pid = os.getpid()
    logger.debug(f"[Worker {pid}] Starting: {case.case_id}")
    start_time = time.time()
    revenue, solve_time_s, status, model_instance = run_optimization(n_scenarios=case.n_scenarios, get_bid_prices_f=case.price_strategy.function, risk_constraint=case.risk_constraint)
    total_time_s = time.time() - start_time
    logger.info(f"[Worker {pid}] Finished: {case.case_id} in {total_time_s:.2f}s (Solve: {f'{solve_time_s:.2f}s' if solve_time_s is not None else 'N/A'}, Status: {status})")
    result = CaseResult(case=case, revenue=revenue, solve_time_s=solve_time_s, total_time_s=total_time_s, status=status, error_message=None)
    if status in ["data_error", "model_error", "execution_error"]: result.error_message = f"Error during {status} phase."
    return result


# --- Parallel Analysis Orchestrator ---
def run_parallel_analysis(cases: List[AnalysisCase], results_filename_base: str) -> List[CaseResult]:
    all_results: List[CaseResult] = []
    incremental_filename = f"{results_filename_base}_incremental.csv"
    logger.info(f"Starting parallel analysis for {len(cases)} cases using {MAX_WORKERS} workers.")
    logger.info(f"Incremental results will be saved to: {incremental_filename}")
    header = ["case_id", "n_scenarios", "price_strategy_desc", "risk_constraint", "status", "revenue", "solve_time_s", "total_time_s", "error_message"]
    try:
        with open(incremental_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(header)
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_case = {executor.submit(_run_single_case_worker, case): case for case in cases}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        result: CaseResult = future.result()
                        all_results.append(result); completed_count += 1
                        logger.debug(f"Completed {completed_count}/{len(cases)}: {case.case_id} (Status: {result.status})")
                        writer.writerow([
                            result.case.case_id, result.case.n_scenarios, result.case.price_strategy.description, result.case.risk_constraint, result.status,
                            f"{result.revenue:.4f}" if result.revenue is not None else "", f"{result.solve_time_s:.4f}" if result.solve_time_s is not None else "",
                            f"{result.total_time_s:.4f}" if result.total_time_s is not None else "", result.error_message if result.error_message else "" ])
                        csvfile.flush()
                    except Exception as exc:
                        completed_count += 1; logger.exception(f"!!! Error processing result future for {case.case_id}: {exc}")
                        error_result = CaseResult(case=case, status="execution_error", error_message=str(exc))
                        all_results.append(error_result)
                        writer.writerow([ error_result.case.case_id, error_result.case.n_scenarios, error_result.case.price_strategy.description, error_result.case.risk_constraint,
                                          error_result.status, "", "", "", error_result.error_message ])
                        csvfile.flush()
    except IOError as e: logger.error(f"Error opening or writing to incremental results file {incremental_filename}: {e}")
    except Exception as e: logger.exception(f"An unexpected error occurred during parallel execution: {e}")
    logger.info(f"Parallel analysis finished. Collected {len(all_results)} results.")
    return all_results


# --- Results Reporting and Visualization ---
def aggregate_results(results: List[CaseResult]) -> Dict[int, Dict[str, CaseResult]]:
    aggregated = {}
    for res in results:
        if res.status not in ["execution_error", "data_error"]:
            n_scen = res.case.n_scenarios; strat_desc = res.case.price_strategy.description
            if n_scen not in aggregated: aggregated[n_scen] = {}
            if strat_desc in aggregated[n_scen]: logger.warning(f"Duplicate result found for {n_scen} scenarios and strategy '{strat_desc}'. Overwriting.")
            aggregated[n_scen][strat_desc] = res
    return aggregated

def generate_summary_table(results: List[CaseResult], output_filename: str):
    logger.info(f"Generating summary table: {output_filename}")
    aggregated_results = aggregate_results(results)
    if not aggregated_results:
        logger.warning("No valid results to generate summary table."); open(output_filename, 'w').write("No valid results found.\n"); return
    scenarios = sorted(aggregated_results.keys())
    strategies = sorted(list(set(strat for scen_results in aggregated_results.values() for strat in scen_results.keys())))
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("Analysis Summary\n" + "=" * 20 + "\n\n"); first_valid_result = next((r for r in results if r.status not in ["execution_error", "data_error"]), None)
            risk_constraint_used = first_valid_result.case.risk_constraint if first_valid_result else "Unknown"; f.write(f"Risk Constraint: {'Enabled' if risk_constraint_used else 'Disabled'}\n\n")
            header = f"{'Strategy':<25}" + "".join([f"{scen:>15}" for scen in scenarios])
            # Revenue Table
            f.write("Expected Revenue ($):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat_desc in strategies:
                f.write(f"{strat_desc:<25}")
                for scen in scenarios:
                    result = aggregated_results.get(scen, {}).get(strat_desc); status_str = result.status if result else 'No Run'
                    if result and result.revenue is not None and result.status in ["optimal", "optimal_inaccurate"]: f.write(f"{result.revenue:>15.2f}")
                    else: f.write(f"{status_str:>15}")
                f.write("\n")
            # Solve Time Table
            f.write("\n\nSolve Time (s):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat_desc in strategies:
                f.write(f"{strat_desc:<25}")
                for scen in scenarios:
                     result = aggregated_results.get(scen, {}).get(strat_desc); status_str = result.status if result else 'No Run'
                     if result and result.solve_time_s is not None and result.status not in ['data_error', 'execution_error']: f.write(f"{result.solve_time_s:>15.2f}")
                     else: f.write(f"{status_str:>15}")
                f.write("\n")
            # Total Worker Time Table
            f.write("\n\nTotal Worker Time (s):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat_desc in strategies:
                f.write(f"{strat_desc:<25}")
                for scen in scenarios:
                     result = aggregated_results.get(scen, {}).get(strat_desc); status_str = result.status if result else 'No Run'
                     if result and result.total_time_s is not None and result.status not in ['data_error', 'execution_error']: f.write(f"{result.total_time_s:>15.2f}")
                     else: f.write(f"{status_str:>15}")
                f.write("\n")
            f.write("\n" + "=" * 20 + "\n")
        logger.info(f"Summary table written to {output_filename}")
    except IOError as e: logger.error(f"Error writing summary table to {output_filename}: {e}")

def plot_results_generic(aggregated_results: Dict[int, Dict[str, CaseResult]], value_accessor: Callable[[CaseResult], Optional[float]], group_by: str, output_filename: str, title: str, ylabel: str, xlabel: Optional[str] = None):
    logger.info(f"Generating plot: {output_filename}"); plt.figure(figsize=(12, 7)); plot_generated = False
    if group_by == 'scenarios':
        scenarios = sorted(aggregated_results.keys()); strategies = sorted(list(set(strat for scen_results in aggregated_results.values() for strat in scen_results.keys())))
        x_ticks_labels = strategies; x_ticks_pos = list(range(len(strategies)))
        for scen in scenarios:
            plot_data = [];
            for i, strat_desc in enumerate(strategies):
                 result = aggregated_results.get(scen, {}).get(strat_desc)
                 if result and result.status in ["optimal", "optimal_inaccurate"]: value = value_accessor(result);
                 if value is not None: plot_data.append((i, value))
            if plot_data: valid_x_ticks, y_values = zip(*plot_data); plt.plot(valid_x_ticks, y_values, marker='o', linestyle='-', label=f"{scen} Scenarios"); plot_generated = True
        plt.xticks(x_ticks_pos, x_ticks_labels, rotation=45, ha='right'); plt.xlabel(xlabel or "Price Strategy")
    elif group_by == 'strategies':
        scenarios = sorted(aggregated_results.keys()); strategies = sorted(list(set(strat for scen_results in aggregated_results.values() for strat in scen_results.keys())))
        x_ticks_pos = scenarios; x_ticks_labels = [str(s) for s in scenarios]
        for strat_desc in strategies:
            plot_data = [];
            for scen in scenarios:
                result = aggregated_results.get(scen, {}).get(strat_desc)
                if result and result.status in ["optimal", "optimal_inaccurate"]: value = value_accessor(result);
                if value is not None: plot_data.append((scen, value))
            if plot_data: valid_x_ticks, y_values = zip(*plot_data); plt.plot(valid_x_ticks, y_values, marker='o', linestyle='-', label=strat_desc); plot_generated = True
        plt.xlabel(xlabel or "Number of Scenarios"); plt.xticks(x_ticks_pos, x_ticks_labels)
    else: raise ValueError("group_by must be 'scenarios' or 'strategies'")
    if not plot_generated: logger.warning(f"No data plotted for {output_filename}. Skipping save."); plt.close(); return
    plt.title(title); plt.ylabel(ylabel); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    try: plt.savefig(output_filename); logger.info(f"Plot saved to {output_filename}")
    except IOError as e: logger.error(f"Error saving plot to {output_filename}: {e}")
    plt.close()

def plot_tradeoff(aggregated_results: Dict[int, Dict[str, CaseResult]], output_filename: str):
    logger.info(f"Generating tradeoff plot: {output_filename}"); plt.figure(figsize=(12, 7))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']; scenarios = sorted(aggregated_results.keys()); marker_idx = 0; plot_generated = False
    for scen in scenarios:
        points = [];
        for strat_desc, result in aggregated_results.get(scen, {}).items():
            if result and result.status in ["optimal", "optimal_inaccurate"]:
                 if result.solve_time_s is not None and result.revenue is not None: points.append((result.solve_time_s, result.revenue, strat_desc))
        if points:
            plot_generated = True; points.sort(); x_values = [p[0] for p in points]; y_values = [p[1] for p in points]
            marker = markers[marker_idx % len(markers)]; plt.plot(x_values, y_values, marker=marker, linestyle='--', alpha=0.7, label=f"{scen} Scenarios"); marker_idx += 1
            for solve_time, revenue, strat_desc in points: short_desc = strat_desc.replace("Fixed Grid Res $", "FG").replace(" Evenly Spaced", "ES").replace("All Unique", "AU"); plt.annotate(short_desc, (solve_time, revenue), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    if not plot_generated: logger.warning(f"No data plotted for {output_filename}. Skipping save."); plt.close(); return
    plt.title("Revenue vs. Solve Time Tradeoff"); plt.xlabel("Solve Time (seconds)"); plt.ylabel("Expected Revenue ($)")
    plt.legend(title="Scenarios", loc='best'); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    try: plt.savefig(output_filename); logger.info(f"Tradeoff plot saved to {output_filename}")
    except IOError as e: logger.error(f"Error saving tradeoff plot to {output_filename}: {e}")
    plt.close()

def visualize_results(results: List[CaseResult], results_filename_base: str):
    logger.info("Generating visualizations..."); aggregated = aggregate_results(results)
    if not aggregated: logger.warning("No valid results to visualize."); return
    scenarios = sorted(aggregated.keys()); strategies = sorted(list(set(strat for scen_results in aggregated.results.values() for strat in scen_results.keys())))
    plot_configs = [
        {'accessor': lambda r: r.revenue, 'suffix': '_revenue_vs_strategy', 'title': 'Revenue vs. Price Strategy', 'ylabel': 'Expected Revenue ($)'},
        {'accessor': lambda r: r.revenue, 'suffix': '_revenue_vs_scenarios', 'title': 'Revenue vs. Number of Scenarios', 'ylabel': 'Expected Revenue ($)'},
        {'accessor': lambda r: r.solve_time_s, 'suffix': '_solvetime_vs_strategy', 'title': 'Solve Time vs. Price Strategy', 'ylabel': 'Solve Time (s)'},
        {'accessor': lambda r: r.solve_time_s, 'suffix': '_solvetime_vs_scenarios', 'title': 'Solve Time vs. Number of Scenarios', 'ylabel': 'Solve Time (s)'}, ]
    for config in plot_configs:
        group_by = 'strategies' if 'vs_scenarios' in config['suffix'] else 'scenarios'
        plot_results_generic(aggregated_results=aggregated, value_accessor=config['accessor'], group_by=group_by, output_filename=f"{results_filename_base}{config['suffix']}.png", title=config['title'], ylabel=config['ylabel'])
    if len(scenarios) > 0 and len(strategies) > 1: plot_tradeoff(aggregated, f"{results_filename_base}_tradeoff.png")


# --- Main Analysis Pipeline ---
# (Code remains the same as previous version)
def run_analysis_pipeline(scenario_list: List[int], price_strategy_configs: List[Dict[str, Any]], use_risk_constraint: bool, timestamp: str):
    logger.info("Starting Analysis Pipeline..."); risk_label = "risk" if use_risk_constraint else "norisk"; analysis_name = f"analysis_{timestamp}_{risk_label}"; results_filename_base = os.path.join(RESULTS_DIR, analysis_name)
    analysis_cases = create_analysis_cases(scenario_list=scenario_list, price_strategy_configs=price_strategy_configs, use_risk_constraint=use_risk_constraint)
    if not analysis_cases: logger.error("No analysis cases generated. Exiting."); return
    all_results = run_parallel_analysis(analysis_cases, results_filename_base)
    summary_filename = f"{results_filename_base}_summary.txt"; generate_summary_table(all_results, summary_filename)
    visualize_results(all_results, results_filename_base)
    logger.info("-" * 50 + "\nAnalysis Pipeline Complete.\n" + f"Incremental results: {results_filename_base}_incremental.csv\n" + f"Summary table:       {summary_filename}\n" + f"Plots saved with prefix: {results_filename_base}_*.png\n" + "-" * 50)


# --- Single Run Function ---
# (Code remains the same as previous version)
def run_optimization_and_report_results(n_scenarios: int, price_strategy: PriceStrategy, is_risk_constraint: bool):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); risk_label = "risk" if is_risk_constraint else "norisk"; strategy_label = price_strategy.description.replace(' ', '_').replace('$', '').replace('.', 'p'); output_file = os.path.join(RESULTS_DIR, f"single_run_{timestamp}_{n_scenarios}s_{strategy_label}_{risk_label}.log")
    logger.remove(); log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"; console_id = logger.add(sys.stdout, level="INFO", format=log_format); file_id = logger.add(output_file, level="INFO", format=log_format, encoding='utf-8')
    try:
        logger.info(f"Detailed Run Report - Started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 80 + "\nConfiguration:\n" + f"  Scenarios:        {n_scenarios}\n" + f"  Price Strategy:   {price_strategy.description}\n" + f"  Risk Constraint:  {'Enabled' if is_risk_constraint else 'Disabled'}\n" + "=" * 80 + "\n")
        start_time = time.time()
        revenue, solve_time, status, model = run_optimization(n_scenarios, price_strategy.function, is_risk_constraint, verbose_market_data=True)
        end_time = time.time()
        logger.info(f"\n--- Results ---\nStatus:           {status}\nSolve Time:       {f'{solve_time:.2f} seconds' if solve_time is not None else 'N/A'}\nTotal Run Time:   {end_time - start_time:.2f} seconds\nExpected Revenue: {f'${revenue:.2f}' if revenue is not None else 'N/A'}")
        if model and status in ["optimal", "optimal_inaccurate"]:
            try:
                model.postprocess_bids(); logger.info("\n--- Bids ---"); active_hours = []; all_hours = model.hours
                for hour_j in all_hours:
                    hour_bids_data = model.all_bids.get(hour_j, {"sell": [], "buy": []}); sell_bids = hour_bids_data.get("sell", []); buy_bids = hour_bids_data.get("buy", [])
                    if not sell_bids and not buy_bids: continue
                    active_hours.append(hour_j); logger.info(f"\nHour {hour_j}:")
                    if sell_bids: logger.info("  Sell bids:")
                    else: logger.info("  No sell bids")
                    for i, bid in enumerate(sell_bids): logger.info(f"    #{i+1}: DA price=${bid['price']:.2f}/MWh, volume={bid['volume_mw']:.1f} MW")
                    if buy_bids: logger.info("  Buy bids:")
                    else: logger.info("  No buy bids")
                    for i, bid in enumerate(buy_bids): logger.info(f"    #{i+1}: DA price=${bid['price']:.2f}/MWh, volume={bid['volume_mw']:.1f} MW")
                logger.info("\nSummary of Active Hours:")
                if active_hours:
                    for hour in sorted(active_hours):
                         hour_bids_data = model.all_bids.get(hour, {"sell": [], "buy": []}); sell_bids = hour_bids_data.get("sell", []); buy_bids = hour_bids_data.get("buy", [])
                         total_sell_volume = sum(bid["volume_mw"] for bid in sell_bids); total_buy_volume = sum(bid["volume_mw"] for bid in buy_bids)
                         logger.info(f"  Hour {hour}: {len(sell_bids)} sell bid(s) (Tot Vol={total_sell_volume:.1f}MW), {len(buy_bids)} buy bid(s) (Tot Vol={total_buy_volume:.1f}MW)")
                else: logger.info("  No active bids found in any hour.")
            except Exception as post_err: logger.exception(f"\nError during bid postprocessing: {post_err}")
        elif model and model.problem: logger.warning(f"\nBids not generated due to non-optimal status: {model.problem.status}")
        else: logger.warning(f"\nBids not generated (model not available or solve failed early).")
        logger.info(f"\n{'=' * 80}\nReport saved to: {output_file}\n{'=' * 80}")
    except Exception as e: logger.exception(f"Error during detailed report generation: {e}")
    finally: logger.remove(console_id); logger.remove(file_id); logger.add(sys.stderr, level="INFO") # Restore default logger
    return revenue, solve_time


# --- Main Execution ---
# (Code remains the same as previous version, using updated analysis config)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stochastic convergence bidding analysis")
    parser.add_argument("--analysis", action="store_true", help="Run multi-case analysis pipeline.")
    parser.add_argument("--scenarios", type=int, default=500, help="Number of scenarios for single run or analysis.")
    price_group = parser.add_mutually_exclusive_group()
    price_group.add_argument("--fixed-grid-resolution", type=float, metavar='RES', help="Use fixed grid with specified $ resolution (e.g., 5 for $5 steps).") # Allow float
    price_group.add_argument("--num-price-points", type=int, metavar='N', help="Use N evenly spaced price points (sample-based).")
    price_group.add_argument("--use-all-unique", action="store_true", help="Use all unique prices from samples (default if no other strategy selected).")
    parser.add_argument("--no-risk-constraint", action="store_true", help="Disable the risk constraint (CVaR). Default is to use it.")
    parser.add_argument("--analysis-scenarios", type=str, default="500,1000,2000", help="Comma-separated list of scenario counts for analysis (e.g., '500,1000,2000').")
    parser.add_argument("--analysis-resolutions", type=str, default="10,5,2,1", help="Comma-separated list of fixed grid resolutions ($) for analysis (e.g., '10,5,2,1').")
    parser.add_argument("--analysis-no-all-unique", action="store_true", help="Exclude 'All Unique Prices' strategy from the analysis.")
    args = parser.parse_args()

    if not args.analysis:
        if not args.fixed_grid_resolution and not args.num_price_points: args.use_all_unique = True

    risk_constraint = not args.no_risk_constraint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.analysis:
        logger.info("Mode: Running Analysis Pipeline")
        try: analysis_scenario_list = [int(s.strip()) for s in args.analysis_scenarios.split(',')]; assert analysis_scenario_list
        except: logger.error(f"Invalid --analysis-scenarios: '{args.analysis_scenarios}'"); sys.exit(1)
        strategy_configs = []
        if args.analysis_resolutions:
             try:
                 resolutions = [float(r.strip()) for r in args.analysis_resolutions.split(',')]; assert all(res > 0 for res in resolutions) # Allow float, check positive
                 for res in resolutions: strategy_configs.append({"strategy_type": "fixed_resolution", "fixed_resolution": res})
             except: logger.error(f"Invalid --analysis-resolutions: '{args.analysis_resolutions}'"); sys.exit(1)
        else: logger.info("No fixed resolutions specified for analysis.")
        if not args.analysis_no_all_unique: strategy_configs.append({"strategy_type": "all_unique"}); logger.info("Including 'All Unique' price strategy.")
        else: logger.info("Excluding 'All Unique' price strategy.")
        if not strategy_configs: logger.error("No price strategies selected for analysis."); sys.exit(1)
        run_analysis_pipeline(scenario_list=analysis_scenario_list, price_strategy_configs=strategy_configs, use_risk_constraint=risk_constraint, timestamp=timestamp)
    else:
        logger.info("Mode: Running Single Detailed Report")
        try:
            if args.fixed_grid_resolution: assert args.fixed_grid_resolution > 0; strategy = create_price_strategy(strategy_type="fixed_resolution", fixed_resolution=args.fixed_grid_resolution)
            elif args.num_price_points: assert args.num_price_points > 0; strategy = create_price_strategy(strategy_type="evenly_spaced", num_points=args.num_price_points)
            elif args.use_all_unique: strategy = create_price_strategy(strategy_type="all_unique")
            else: logger.error("No price strategy specified for single run."); sys.exit(1)
            run_optimization_and_report_results(n_scenarios=args.scenarios, price_strategy=strategy, is_risk_constraint=risk_constraint)
        except (ValueError, AssertionError) as e: logger.error(f"Invalid argument for price strategy: {e}"); sys.exit(1)
