# --- START OF REVISED run_project.py with Fixed OOS Eval Size & Seeding ---

import time
import numpy as np
import sys
import os
import concurrent.futures
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable, Optional, NamedTuple, Any
from dataclasses import dataclass, field
import csv
import argparse
import xarray as xr
import cvxpy as cp
import hashlib # For seeding
import threading # For thread pool

# --- Loguru Setup ---
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Set max workers for parallel processing
MAX_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4

from stochastic_convergence_bidding.bidding_model import BiddingModel, BidPriceFunction, calculate_empirical_cvar # Import new helper if needed
from stochastic_convergence_bidding.market_data import MarketData
from stochastic_convergence_bidding.sample_data_generator import generate_sample_data

# --- Constants ---
BASE_SEED = 2024 # Base seed for reproducibility
RESULTS_DIR = "./results"
HOURS_LIST = list(range(24)) # Define default hours list
N_OOS_SCENARIOS = 5000 # <--- FIXED SIZE for Out-of-Sample evaluation dataset

# Create output directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Caching and Seeding ---
# Cache Key: (thread_id, n_scenarios, purpose) -> Value: (data, target_names)
sample_data_cache: Dict[Tuple[int, int, str], Tuple[np.ndarray, List[str]]] = {}

def get_seed(n_scenarios: int, purpose: str) -> int:
    """Generates a deterministic seed based on BASE_SEED, scenario count, and purpose."""
    seed_str = f"{BASE_SEED}-{n_scenarios}-{purpose}"
    hash_obj = hashlib.sha256(seed_str.encode())
    seed_int = int.from_bytes(hash_obj.digest()[:4], 'big')
    return seed_int % (2**32 - 1)

def get_or_generate_data(n_scenarios: int, purpose: str, thread_id: int) -> Tuple[np.ndarray, List[str]]:
    """Gets data from cache or generates it using a deterministic seed."""
    cache_key = (thread_id, n_scenarios, purpose)
    if cache_key in sample_data_cache:
        # logger.debug(f"Cache hit for {n_scenarios} scenarios ({purpose}) on thread {thread_id}")
        return sample_data_cache[cache_key]
    else:
        # logger.debug(f"Cache miss for {n_scenarios} scenarios ({purpose}) on thread {thread_id}. Generating...")
        seed = get_seed(n_scenarios, purpose)
        # logger.debug(f"Generating data: n={n_scenarios}, purpose={purpose}, seed={seed}")
        data, target_names = generate_sample_data(
            num_samples=n_scenarios,
            num_hours=max(HOURS_LIST) + 1,
            random_seed=seed
        )
        sample_data_cache[cache_key] = (data, target_names)
        return data, target_names

# --- Price Strategy Implementations ---
def use_all_unique_prices() -> BidPriceFunction:
    """Creates a function returning unique DA prices from the dataset."""
    def get_prices(hour_data: xr.Dataset) -> np.ndarray:
        if "dalmp" not in hour_data: logger.error("'dalmp' not found"); return np.array([])
        prices = hour_data["dalmp"].values
        return np.unique(prices[~np.isnan(prices)])
    return get_prices

def evenly_spaced_sample_based_prices(num_bid_prices: int) -> BidPriceFunction:
    """Creates a function returning evenly spaced prices."""
    if num_bid_prices <= 0: raise ValueError("num_bid_prices must be positive")
    def get_prices(hour_data: xr.Dataset) -> np.ndarray:
        if "dalmp" not in hour_data: logger.error("'dalmp' not found"); return np.array([])
        prices = hour_data["dalmp"].values; valid_prices = prices[~np.isnan(prices)]
        if valid_prices.size == 0: logger.warning(f"No valid prices hour {hour_data['hour'].item()}"); return np.array([])
        min_p, max_p = np.min(valid_prices), np.max(valid_prices)
        if min_p == max_p: return np.array([min_p])
        return np.linspace(min_p, max_p, max(1, num_bid_prices))
    return get_prices

def fixed_grid_prices(price_grid: List[float]) -> BidPriceFunction:
    """Creates a function returning a fixed grid of prices."""
    np_price_grid = np.array(price_grid, dtype=np.float64)
    if np_price_grid.size == 0: logger.warning("fixed_grid_prices created with empty grid.")
    def get_prices(hour_data: xr.Dataset) -> np.ndarray: return np_price_grid
    return get_prices

def subsample_every_n(
        stride: int = 1,
        round_decimals: Optional[int] = None
) -> BidPriceFunction:
    """
    Creates a function that returns unique DA prices from the dataset,
    optionally rounded, and then subsampled using a stride.
    (Assumes input data is clean: 'dalmp' exists, no NaNs).

    Args:
        stride: Take every 'stride'-th unique price after sorting.
                stride=1 means take all unique prices. Must be > 0.
        round_decimals: If not None, round the DA prices to this many
                        decimal places BEFORE finding unique values.

    Returns:
        A BidPriceFunction.

    Raises:
        ValueError: If stride is invalid or if the process results in zero prices.
    """
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    if round_decimals is not None and round_decimals < 0:
        raise ValueError("round_decimals must be None or a non-negative integer.")

    def get_prices(hour_data: xr.Dataset) -> np.ndarray:
        # --- Streamlined Core Logic ---
        # Directly access and process, assuming data validity
        sampled_prices = hour_data["dalmp"].values

        # 1. Optional Rounding
        if round_decimals is not None:
            processed_prices = np.round(sampled_prices, decimals=round_decimals)
        else:
            processed_prices = sampled_prices

        # 2. Get Unique Sorted Prices
        unique_prices = np.unique(processed_prices)
        final_prices = unique_prices[::stride]  # Handles stride=1 automatically

        if final_prices.size < 4:
            logger.warning(f"Price generation resulted in only {final_prices.size} prices for hour {hour_data['hour'].item()} (stride={stride}).")

        return final_prices

    return get_prices  # Return the inner function


# --- Data Structures ---
@dataclass(frozen=True)
class PriceStrategy:
    """Encapsulates a function to get bid prices, its description, and properties."""
    function: BidPriceFunction
    description: str
    is_fixed_grid: bool
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AnalysisCase:
    """Parameters defining a single analysis run."""
    case_id: str
    n_scenarios: int # This is the TRAINING scenario count
    price_strategy: PriceStrategy
    risk_constraint: bool

@dataclass
class CaseResult:
    """Results from a single analysis run, including OOS evaluation."""
    case: AnalysisCase
    fit_revenue: Optional[float] = None
    fit_cvar: Optional[float] = None
    fit_status: Optional[str] = None
    total_worker_time_s: Optional[float] = None
    oos_revenue: Optional[float] = None
    oos_cvar: Optional[float] = None
    oos_eval_status: Optional[str] = None
    error_message: Optional[str] = None

# --- Price Strategy Factory ---
def create_price_strategy(
    strategy_type: str, num_points: Optional[int] = None, fixed_resolution: Optional[float] = None,
    fixed_grid_list: Optional[List[float]] = None, min_price: float = -100.0, max_price: float = 200.0,
    stride: Optional[int] = None, round_decimals: Optional[int] = None
) -> PriceStrategy:
    """Creates a PriceStrategy object, including the is_fixed_grid flag."""
    base_config = locals(); is_fixed = False
    
    # Extra parameters specific to subsample strategy
    stride = base_config.pop("stride", None)
    round_decimals = base_config.pop("round_decimals", None)
    
    if strategy_type == "all_unique":
        func = use_all_unique_prices(); desc = "All Unique"
    elif strategy_type == "evenly_spaced":
        if num_points is None: raise ValueError("num_points missing");
        func = evenly_spaced_sample_based_prices(num_points); desc = f"{num_points} Evenly Spaced"
    elif strategy_type == "fixed_resolution":
        if fixed_resolution is None: raise ValueError("fixed_resolution missing");
        if fixed_resolution <= 0: raise ValueError("fixed_resolution must be positive")
        grid = list(np.arange(min_price, max_price + fixed_resolution/2.0, fixed_resolution))
        func = fixed_grid_prices(grid); desc = f"Fixed Grid Res ${fixed_resolution:.2f}"; is_fixed = True # Format resolution
        base_config['generated_grid_list'] = grid
    elif strategy_type == "fixed_list":
        if fixed_grid_list is None: raise ValueError("fixed_grid_list missing");
        func = fixed_grid_prices(fixed_grid_list); desc = "Fixed Grid Custom List"; is_fixed = True
    elif strategy_type == "subsample_every_n":
        if stride is None:
            stride = 1  # Default stride
        if stride <= 0: 
            raise ValueError("stride must be positive")
        
        func = subsample_every_n(stride=stride, round_decimals=round_decimals)
        desc = f"Every {stride}th Price"
        if round_decimals is not None:
            desc += f", rounded to {round_decimals} decimal{'s' if round_decimals != 1 else ''}"
        
        # Store special parameters in config
        base_config["stride"] = stride
        if round_decimals is not None:
            base_config["round_decimals"] = round_decimals
            
        # This is NOT a fixed grid price strategy (depends on the data)
        is_fixed = False
        
    else: raise ValueError(f"Unknown strategy_type: {strategy_type}")
    return PriceStrategy(function=func, description=desc, config=base_config, is_fixed_grid=is_fixed)

# --- Analysis Case Generation ---
def create_analysis_cases(
    scenario_list: List[int], price_strategy_configs: List[Dict[str, Any]], use_risk_constraint: bool
) -> List[AnalysisCase]:
    """Generates AnalysisCase objects. case.n_scenarios is for training."""
    cases = []
    case_counter = 0
    for n_scenarios_train in scenario_list: # Use explicit name
        for strategy_config in price_strategy_configs:
            try:
                price_strategy = create_price_strategy(**strategy_config)
                if price_strategy.config.get('strategy_type') == 'evenly_spaced':
                    num_points = price_strategy.config.get('num_points')
                    if num_points is not None and num_points >= n_scenarios_train:
                        logger.info(f"Skipping case: {n_scenarios_train} train scenarios, {price_strategy.description} (num_points >= n_scenarios_train)")
                        continue
                risk_label = 'risk' if use_risk_constraint else 'norisk'
                strat_label = price_strategy.description.replace(' ', '_').replace('$', '').replace('.', 'p')
                case_id = f"case_{case_counter:04d}_{n_scenarios_train}s_{strat_label}_{risk_label}"
                case = AnalysisCase(case_id=case_id, n_scenarios=n_scenarios_train, price_strategy=price_strategy, risk_constraint=use_risk_constraint)
                cases.append(case); case_counter += 1
            except ValueError as e: logger.warning(f"Skipping case generation: {e}")
            except Exception as e: logger.exception(f"Error generating case for {strategy_config}: {e}")
    logger.info(f"Generated {len(cases)} analysis cases.")
    return cases

# --- Single Case Runner (Fit ONLY - for single run mode) ---
def fit_single_case(
    n_scenarios: int, get_bid_prices_f: BidPriceFunction, risk_constraint: bool,
    alpha: float = 0.95, rho: float = -1000.0, verbose_market_data: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[BiddingModel]]:
    """Fits the model for a single case (used by detailed report function)."""
    thread_id = threading.get_ident()
    try:
        # Only needs training data
        train_data_arr, target_names = get_or_generate_data(n_scenarios, "train", thread_id)
        train_market_data = MarketData(train_data_arr, target_names)
        if verbose_market_data: train_market_data.print_hourly_report()
    except Exception as e:
        logger.error(f"[Thread {thread_id}] Error getting training data for {n_scenarios}: {e}")
        return None, None, "data_error", None

    model = BiddingModel(
        train_market_data=train_market_data, hours=HOURS_LIST,
        get_bid_prices_f=get_bid_prices_f, alpha=alpha, rho=rho, verbose_solver=False)

    start_fit_time = time.time() # Time the entire fit process
    try:
        model.fit(risk_constraint=risk_constraint, solver=cp.CLARABEL)
    except Exception as e:
        logger.exception(f"[Thread {thread_id}] Error during model.fit() for {n_scenarios}: {e}")
        fit_time_s = time.time() - start_fit_time
        # Return approximate time even on error, adjust status
        return None, model.fit_cvar_value, "fit_error", model
    fit_time_s = time.time() - start_fit_time

    # Return fit results including IS CVaR and total fit time
    return model.fit_objective_value, model.fit_cvar_value, model.fit_status, fit_time_s, model


# --- Parallel Execution Worker (Fit + Evaluate OOS) ---
def _run_single_case_worker(case: AnalysisCase) -> CaseResult:
    """ Worker function: fits model, evaluates OOS if applicable. """
    thread_id = threading.get_ident(); thread_name = threading.current_thread().name
    logger.debug(f"[Thread {thread_name}] Starting: {case.case_id} (Train N={case.n_scenarios}, OOS N={N_OOS_SCENARIOS})")
    start_time = time.time()

    fit_revenue, fit_cvar, fit_status = None, None, None
    oos_revenue, oos_cvar, oos_eval_status = None, None, "not_run"
    model_instance = None; error_message = None

    try:
        # 1. Get Data (Train uses case.n_scenarios, Test uses N_OOS_SCENARIOS)
        train_data_arr, target_names = get_or_generate_data(case.n_scenarios, "train", thread_id)
        train_market_data = MarketData(train_data_arr, target_names)

        test_data_arr, _ = get_or_generate_data(N_OOS_SCENARIOS, "test", thread_id) # Use constant
        test_market_data = MarketData(test_data_arr, target_names)

        # 2. Instantiate Model
        model_instance = BiddingModel(train_market_data=train_market_data, hours=HOURS_LIST, get_bid_prices_f=case.price_strategy.function)

        # 3. Fit Model
        model_instance.fit(risk_constraint=case.risk_constraint, solver=cp.CLARABEL)
        fit_revenue = model_instance.fit_objective_value
        fit_cvar = model_instance.fit_cvar_value
        fit_status = model_instance.fit_status

        # 4. Evaluate OOS (Conditional)
        if fit_status in ["optimal", "optimal_inaccurate"]:
            if case.price_strategy.is_fixed_grid:
                try:
                    oos_revenue, oos_cvar, oos_eval_status = model_instance.evaluate_oos(test_market_data)
                except Exception as oos_err:
                    logger.exception(f"[Thread {thread_name}] Error during evaluate_oos call for {case.case_id}: {oos_err}")
                    oos_eval_status = "eval_exception"; error_message = f"OOS Eval Exception: {oos_err}"
            else: oos_eval_status = "skipped_non_fixed_grid"
        else: oos_eval_status = "skipped_fit_failed"

    except Exception as worker_err:
        logger.exception(f"[Thread {thread_name}] Error during worker execution for {case.case_id}: {worker_err}")
        fit_status = fit_status or "worker_exception"; error_message = f"Worker Exception: {worker_err}"

    total_worker_time_s = time.time() - start_time
    logger.info(f"[Thread {thread_name}] Finished: {case.case_id} in {total_worker_time_s:.2f}s "
                f"(Fit: {fit_status}, OOS: {oos_eval_status})")

    result = CaseResult(
        case=case, fit_revenue=fit_revenue, fit_cvar=fit_cvar, fit_status=fit_status,
        total_worker_time_s=total_worker_time_s, oos_revenue=oos_revenue, oos_cvar=oos_cvar,
        oos_eval_status=oos_eval_status, error_message=error_message )
    return result


# --- Parallel Analysis Orchestrator ---
def run_parallel_analysis(cases: List[AnalysisCase], results_filename_base: str) -> List[CaseResult]:
    """ Runs analysis cases in parallel (Threads), collects results, saves incrementally. """
    all_results: List[CaseResult] = []
    incremental_filename = f"{results_filename_base}_incremental.csv"
    logger.info(f"Starting parallel analysis for {len(cases)} cases using {MAX_WORKERS} threads.")
    logger.info(f"Incremental results will be saved to: {incremental_filename}")
    header = [
        "case_id", "n_scenarios_train", "price_strategy_desc", "is_fixed_grid", "risk_constraint", # Added _train suffix
        "fit_status", "fit_revenue", "fit_cvar", "total_worker_time_s",
        "oos_eval_status", "oos_revenue", "oos_cvar", "error_message" ]
    try:
        with open(incremental_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(header)
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="BiddingWorker") as executor:
                future_to_case = {executor.submit(_run_single_case_worker, case): case for case in cases}
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        result: CaseResult = future.result()
                        all_results.append(result); completed_count += 1
                        logger.debug(f"Completed {completed_count}/{len(cases)}: {case.case_id} (Fit: {result.fit_status}, OOS: {result.oos_eval_status})")
                        writer.writerow([
                            result.case.case_id, result.case.n_scenarios, result.case.price_strategy.description, # n_scenarios is training N
                            result.case.price_strategy.is_fixed_grid, result.case.risk_constraint,
                            result.fit_status, f"{result.fit_revenue:.4f}" if result.fit_revenue is not None else "",
                            f"{result.fit_cvar:.4f}" if result.fit_cvar is not None else "",
                            f"{result.total_worker_time_s:.4f}" if result.total_worker_time_s is not None else "",
                            result.oos_eval_status, f"{result.oos_revenue:.4f}" if result.oos_revenue is not None else "",
                            f"{result.oos_cvar:.4f}" if result.oos_cvar is not None else "",
                            result.error_message if result.error_message else "" ])
                        csvfile.flush()
                    except Exception as exc:
                        completed_count += 1; logger.exception(f"!!! Error processing result future for {case.case_id}: {exc}")
                        error_result = CaseResult(case=case, fit_status="execution_error", error_message=str(exc))
                        all_results.append(error_result)
                        writer.writerow([ error_result.case.case_id, error_result.case.n_scenarios, error_result.case.price_strategy.description,
                                          error_result.case.price_strategy.is_fixed_grid, error_result.case.risk_constraint,
                                          error_result.fit_status, "", "", "", "", "", "", error_result.error_message ])
                        csvfile.flush()
    except IOError as e: logger.error(f"Error accessing incremental results file {incremental_filename}: {e}")
    except Exception as e: logger.exception(f"Unexpected error during parallel execution: {e}")
    logger.info(f"Parallel analysis finished. Collected {len(all_results)} results.")
    return all_results


# --- Results Reporting and Visualization ---
def aggregate_results(results: List[CaseResult]) -> Dict[int, Dict[str, CaseResult]]:
    """ Aggregates results by training scenario count and strategy description. """
    aggregated = {}
    for res in results:
        if res.fit_status not in ["execution_error", "data_error", "worker_exception"]:
            n_scen_train = res.case.n_scenarios # Use training N for aggregation key
            strat_desc = res.case.price_strategy.description
            if n_scen_train not in aggregated: aggregated[n_scen_train] = {}
            if strat_desc in aggregated[n_scen_train]: logger.warning(f"Duplicate result: {n_scen_train} train scenarios, '{strat_desc}'. Overwriting.")
            aggregated[n_scen_train][strat_desc] = res
    return aggregated

def generate_summary_table(results: List[CaseResult], output_filename: str):
    """ Generates summary table including OOS results. """
    logger.info(f"Generating summary table: {output_filename}")
    aggregated_results = aggregate_results(results)
    if not aggregated_results:
        logger.warning("No valid results for summary table."); open(output_filename, 'w').write("No valid results.\n"); return
    train_scenarios = sorted(aggregated_results.keys()) # Now using training N
    strategies = sorted(list(set(strat for r in aggregated_results.values() for strat in r.keys())))
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("Analysis Summary\n" + "=" * 20 + "\n\n"); first_valid = next((r for r in results if r.fit_status), None)
            risk_used = first_valid.case.risk_constraint if first_valid else "Unknown"; f.write(f"Risk Constraint: {'Enabled' if risk_used else 'Disabled'}\n")
            f.write(f"OOS Evaluation Scenarios: {N_OOS_SCENARIOS}\n\n") # State the fixed OOS N
            header = f"{'Strategy':<25}" + "".join([f"{scen:>15}" for scen in train_scenarios]) # Header uses training N
            f.write("Columns show results based on TRAINING scenario count.\n")

            # IS Revenue
            f.write("\nIn-Sample Expected Revenue ($):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat in strategies:
                f.write(f"{strat:<25}")
                for scen in train_scenarios: # Iterate through training N
                    res = aggregated_results.get(scen, {}).get(strat); status = res.fit_status if res else 'No Run'
                    if res and res.fit_revenue is not None and res.fit_status in ["optimal", "optimal_inaccurate"]: f.write(f"{res.fit_revenue:>15.2f}")
                    else: f.write(f"{status:>15}")
                f.write("\n")

            # IS CVaR
            f.write("\n\nIn-Sample CVaR ($):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat in strategies:
                f.write(f"{strat:<25}")
                for scen in train_scenarios:
                    res = aggregated_results.get(scen, {}).get(strat); status = res.fit_status if res else 'No Run'
                    if res and res.fit_cvar is not None and res.fit_status in ["optimal", "optimal_inaccurate"]: f.write(f"{res.fit_cvar:>15.2f}")
                    else: f.write(f"{status:>15}")
                f.write("\n")

            # OOS Revenue
            f.write("\n\nOut-of-Sample Expected Revenue ($):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat in strategies:
                f.write(f"{strat:<25}")
                for scen in train_scenarios:
                    res = aggregated_results.get(scen, {}).get(strat); status = res.oos_eval_status if res else 'No Run'
                    if res and res.oos_revenue is not None and res.oos_eval_status == "success": f.write(f"{res.oos_revenue:>15.2f}")
                    else: f.write(f"{status:>15}")
                f.write("\n")

            # OOS CVaR
            f.write("\n\nOut-of-Sample CVaR ($):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat in strategies:
                f.write(f"{strat:<25}")
                for scen in train_scenarios:
                    res = aggregated_results.get(scen, {}).get(strat); status = res.oos_eval_status if res else 'No Run'
                    if res and res.oos_cvar is not None and res.oos_eval_status == "success": f.write(f"{res.oos_cvar:>15.2f}")
                    else: f.write(f"{status:>15}")
                f.write("\n")

            # Worker Time
            f.write("\n\nTotal Worker Time (s):\n"); f.write(header + "\n" + "-" * len(header) + "\n")
            for strat in strategies:
                f.write(f"{strat:<25}")
                for scen in train_scenarios:
                     res = aggregated_results.get(scen, {}).get(strat); status = res.fit_status if res else 'No Run'
                     if res and res.total_worker_time_s is not None: f.write(f"{res.total_worker_time_s:>15.2f}")
                     else: f.write(f"{status:>15}")
                f.write("\n")

            f.write("\n" + "=" * 20 + "\n")
        logger.info(f"Summary table written to {output_filename}")
    except IOError as e: logger.error(f"Error writing summary table: {e}")

def plot_results_generic(aggregated_results: Dict[int, Dict[str, CaseResult]], value_accessor: Callable[[CaseResult], Optional[float]], group_by: str, output_filename: str, title: str, ylabel: str, xlabel: Optional[str] = None, status_checker: Optional[Callable[[CaseResult], bool]] = None):
    """ Generic plotting function. Aggregation keys are training N. """
    logger.info(f"Generating plot: {output_filename}"); plt.figure(figsize=(12, 7)); plot_generated = False
    if status_checker is None: status_checker = lambda r: r.fit_status in ["optimal", "optimal_inaccurate"] # Default check fit status

    if group_by == 'scenarios': # Plot vs Strategy, lines grouped by Training N
        train_scenarios = sorted(aggregated_results.keys())
        strategies = sorted(list(set(strat for r in aggregated_results.values() for strat in r.keys())))
        x_ticks_labels = strategies; x_ticks_pos = list(range(len(strategies)))
        for scen in train_scenarios:
            plot_data = []
            for i, strat_desc in enumerate(strategies):
                 result = aggregated_results.get(scen, {}).get(strat_desc)
                 if result and status_checker(result): # Apply status check
                    value = value_accessor(result)
                    if value is not None: plot_data.append((i, value))
            if plot_data: valid_x_ticks, y_values = zip(*plot_data); plt.plot(valid_x_ticks, y_values, marker='o', linestyle='-', label=f"{scen} Train Scen"); plot_generated = True # Label line by training N
        plt.xticks(x_ticks_pos, x_ticks_labels, rotation=45, ha='right'); plt.xlabel(xlabel or "Price Strategy")

    elif group_by == 'strategies': # Plot vs Training N, lines grouped by Strategy
        train_scenarios = sorted(aggregated_results.keys())
        strategies = sorted(list(set(strat for r in aggregated_results.values() for strat in r.keys())))
        x_ticks_pos = train_scenarios; x_ticks_labels = [str(s) for s in train_scenarios]
        for strat_desc in strategies:
            plot_data = []
            for scen in train_scenarios:
                result = aggregated_results.get(scen, {}).get(strat_desc)
                if result and status_checker(result): # Apply status check
                    value = value_accessor(result)
                    if value is not None: plot_data.append((scen, value))
            if plot_data: valid_x_ticks, y_values = zip(*plot_data); plt.plot(valid_x_ticks, y_values, marker='o', linestyle='-', label=strat_desc); plot_generated = True
        plt.xlabel(xlabel or "Number of Training Scenarios"); plt.xticks(x_ticks_pos, x_ticks_labels) # X-axis is training N

    else: raise ValueError("group_by must be 'scenarios' or 'strategies'")

    if not plot_generated: logger.warning(f"No data plotted for {output_filename}. Skipping save."); plt.close(); return
    plt.title(title); plt.ylabel(ylabel); plt.legend(); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    try: plt.savefig(output_filename); logger.info(f"Plot saved to {output_filename}")
    except IOError as e: logger.error(f"Error saving plot: {e}")
    plt.close()

def plot_is_oos_metric_comparison(aggregated_results: Dict[int, Dict[str, CaseResult]], metric_accessor_is: Callable, metric_accessor_oos: Callable, metric_name: str, status_checker: Callable[[CaseResult], bool], output_filename: str):
    """ Plots IS vs OOS for a metric (Revenue or CVaR), grouped by Training N. """
    logger.info(f"Generating IS vs OOS plot for {metric_name}: {output_filename}")
    plt.figure(figsize=(14, 8)); plot_generated = False
    train_scenarios = sorted(aggregated_results.keys())
    # Strategies eligible for OOS comparison
    strategies = sorted([s for s, r in aggregated_results.get(train_scenarios[0], {}).items() if r.case.price_strategy.is_fixed_grid])
    if not strategies: logger.warning(f"No fixed grid strategies for IS/OOS {metric_name} plot."); plt.close(); return

    num_strategies = len(strategies); num_train_scenarios = len(train_scenarios)
    bar_width_total = 0.8; bar_width_scenario = bar_width_total / num_train_scenarios
    index = np.arange(num_strategies)
    colors = plt.cm.get_cmap('tab10', num_train_scenarios)

    for i, scen in enumerate(train_scenarios):
        is_metrics, oos_metrics, valid_strat_indices = [], [], []
        for j, strat_desc in enumerate(strategies):
            result = aggregated_results.get(scen, {}).get(strat_desc)
            if result and status_checker(result): # Check if valid for OOS plotting
                is_metric_val = metric_accessor_is(result)
                oos_metric_val = metric_accessor_oos(result)
                if is_metric_val is not None and oos_metric_val is not None:
                    is_metrics.append(is_metric_val); oos_metrics.append(oos_metric_val)
                    valid_strat_indices.append(j); plot_generated = True

        if is_metrics:
            group_offset = (i - (num_train_scenarios - 1) / 2) * bar_width_scenario
            current_indices = index[valid_strat_indices]
            current_is_metrics = [is_metrics[k] for k, idx in enumerate(valid_strat_indices)]
            current_oos_metrics = [oos_metrics[k] for k, idx in enumerate(valid_strat_indices)]

            plt.bar(current_indices + group_offset - bar_width_scenario/4, current_is_metrics, bar_width_scenario / 2, label=f'{scen} Trn Scen IS' if i == 0 else "", color=colors(i), hatch='//')
            plt.bar(current_indices + group_offset + bar_width_scenario/4, current_oos_metrics, bar_width_scenario / 2, label=f'{scen} Trn Scen OOS' if i == 0 else "", color=colors(i))

    if not plot_generated: logger.warning(f"No valid IS/OOS data plotted for {metric_name} ({output_filename})."); plt.close(); return
    plt.xlabel('Price Strategy (Fixed Grid Only)'); plt.ylabel(f'{metric_name} ($)')
    plt.title(f'In-Sample vs. Out-of-Sample {metric_name} Comparison (Grouped by Training N)')
    plt.xticks(index, strategies, rotation=45, ha='right')
    from matplotlib.patches import Patch # For custom legend
    legend_elements = []
    for i, scen in enumerate(train_scenarios):
         legend_elements.append(Patch(facecolor=colors(i), hatch='//', label=f'{scen} Train IS'))
         legend_elements.append(Patch(facecolor=colors(i), label=f'{scen} Train OOS'))
    plt.legend(handles=legend_elements, title="Train Scenarios / Type", loc='best', ncol=max(1, num_train_scenarios // 2))
    plt.grid(True, axis='y', linestyle='--', alpha=0.6); plt.tight_layout()
    try: plt.savefig(output_filename); logger.info(f"Plot saved to {output_filename}")
    except IOError as e: logger.error(f"Error saving IS/OOS {metric_name} plot: {e}")
    plt.close()


def visualize_results(results: List[CaseResult], results_filename_base: str):
    """ Generates plots including OOS results and comparisons. """
    logger.info("Generating visualizations..."); aggregated = aggregate_results(results)
    if not aggregated: logger.warning("No valid results to visualize."); return
    train_scenarios = sorted(aggregated.keys()); # Keys are now training N
    strategies = sorted(list(set(strat for r in aggregated.values() for strat in r.keys())))
    is_status_check = lambda r: r.fit_status in ["optimal", "optimal_inaccurate"]
    oos_status_check = lambda r: r.oos_eval_status == "success" # Check OOS specific status

    # Plot IS Revenue
    plot_results_generic(aggregated, value_accessor=lambda r: r.fit_revenue, group_by='scenarios', output_filename=f"{results_filename_base}_ISrevenue_vs_strategy.png", title='In-Sample Revenue vs. Price Strategy', ylabel='IS Expected Revenue ($)', status_checker=is_status_check)
    plot_results_generic(aggregated, value_accessor=lambda r: r.fit_revenue, group_by='strategies', output_filename=f"{results_filename_base}_ISrevenue_vs_scenarios.png", title='In-Sample Revenue vs. Number of Training Scenarios', ylabel='IS Expected Revenue ($)', status_checker=is_status_check)
    # Plot IS CVaR
    plot_results_generic(aggregated, value_accessor=lambda r: r.fit_cvar, group_by='scenarios', output_filename=f"{results_filename_base}_ISCVaR_vs_strategy.png", title='In-Sample CVaR vs. Price Strategy', ylabel=f'IS CVaR ($)', status_checker=is_status_check)
    plot_results_generic(aggregated, value_accessor=lambda r: r.fit_cvar, group_by='strategies', output_filename=f"{results_filename_base}_ISCVaR_vs_scenarios.png", title='In-Sample CVaR vs. Number of Training Scenarios', ylabel=f'IS CVaR ($)', status_checker=is_status_check)

    # Plot OOS Revenue
    plot_results_generic(aggregated, value_accessor=lambda r: r.oos_revenue, group_by='scenarios', output_filename=f"{results_filename_base}_OOSrevenue_vs_strategy.png", title='Out-of-Sample Revenue vs. Price Strategy', ylabel='OOS Expected Revenue ($)', status_checker=oos_status_check)
    plot_results_generic(aggregated, value_accessor=lambda r: r.oos_revenue, group_by='strategies', output_filename=f"{results_filename_base}_OOSrevenue_vs_scenarios.png", title='Out-of-Sample Revenue vs. Number of Training Scenarios', ylabel='OOS Expected Revenue ($)', status_checker=oos_status_check)
    # Plot OOS CVaR
    plot_results_generic(aggregated, value_accessor=lambda r: r.oos_cvar, group_by='scenarios', output_filename=f"{results_filename_base}_OOSCVaR_vs_strategy.png", title='Out-of-Sample CVaR vs. Price Strategy', ylabel=f'OOS CVaR ($)', status_checker=oos_status_check)
    plot_results_generic(aggregated, value_accessor=lambda r: r.oos_cvar, group_by='strategies', output_filename=f"{results_filename_base}_OOSCVaR_vs_scenarios.png", title='Out-of-Sample CVaR vs. Number of Training Scenarios', ylabel=f'OOS CVaR ($)', status_checker=oos_status_check)

    # Plot Worker Time
    plot_results_generic(aggregated, value_accessor=lambda r: r.total_worker_time_s, group_by='scenarios', output_filename=f"{results_filename_base}_workertime_vs_strategy.png", title='Total Worker Time vs. Price Strategy', ylabel='Worker Time (s)')
    plot_results_generic(aggregated, value_accessor=lambda r: r.total_worker_time_s, group_by='strategies', output_filename=f"{results_filename_base}_workertime_vs_scenarios.png", title='Total Worker Time vs. Number of Training Scenarios', ylabel='Worker Time (s)')

    # Plot Tradeoff (IS Revenue vs Worker Time)
    if len(train_scenarios) > 0 and len(strategies) > 1: plot_tradeoff(aggregated, f"{results_filename_base}_tradeoff_IS_Revenue.png")

    # Plot IS vs OOS comparisons (only for successful OOS cases)
    plot_is_oos_metric_comparison(aggregated, metric_accessor_is=lambda r: r.fit_revenue, metric_accessor_oos=lambda r: r.oos_revenue, metric_name="Revenue", status_checker=oos_status_check, output_filename=f"{results_filename_base}_IS_vs_OOS_revenue.png")
    plot_is_oos_metric_comparison(aggregated, metric_accessor_is=lambda r: r.fit_cvar, metric_accessor_oos=lambda r: r.oos_cvar, metric_name="CVaR", status_checker=oos_status_check, output_filename=f"{results_filename_base}_IS_vs_OOS_CVaR.png")


# --- Main Analysis Pipeline ---
def run_analysis_pipeline(scenario_list: List[int], price_strategy_configs: List[Dict[str, Any]], use_risk_constraint: bool, timestamp: str):
    """ Orchestrates the analysis including OOS evaluation. """
    logger.info("Starting Analysis Pipeline..."); risk_label = "risk" if use_risk_constraint else "norisk"; analysis_name = f"analysis_{timestamp}_{risk_label}"; results_filename_base = os.path.join(RESULTS_DIR, analysis_name)
    analysis_cases = create_analysis_cases(scenario_list=scenario_list, price_strategy_configs=price_strategy_configs, use_risk_constraint=use_risk_constraint)
    if not analysis_cases: logger.error("No analysis cases generated. Exiting."); return
    all_results = run_parallel_analysis(analysis_cases, results_filename_base)
    summary_filename = f"{results_filename_base}_summary.txt"; generate_summary_table(all_results, summary_filename)
    visualize_results(all_results, results_filename_base)
    logger.info("-" * 50 + "\nAnalysis Pipeline Complete.\n" + f"OOS Evaluation Scenarios: {N_OOS_SCENARIOS}\n" + f"Incremental results: {results_filename_base}_incremental.csv\n" + f"Summary table:       {summary_filename}\n" + f"Plots saved with prefix: {results_filename_base}_*.png\n" + "-" * 50)


# --- Single Run Function (No OOS) ---
def run_optimization_and_report_results(n_scenarios: int, price_strategy: PriceStrategy, is_risk_constraint: bool):
    """ Runs a single case (fit only) and reports details. """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); risk_label = "risk" if is_risk_constraint else "norisk"; strategy_label = price_strategy.description.replace(' ', '_').replace('$', '').replace('.', 'p'); output_file = os.path.join(RESULTS_DIR, f"single_run_{timestamp}_{n_scenarios}s_{strategy_label}_{risk_label}.log")
    logger.remove(); log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"; console_id = logger.add(sys.stdout, level="INFO", format=log_format); file_id = logger.add(output_file, level="INFO", format=log_format, encoding='utf-8')
    try:
        logger.info(f"Detailed Run Report - Started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "=" * 80 + "\nConfiguration:\n" + f"  Scenarios:        {n_scenarios}\n" + f"  Price Strategy:   {price_strategy.description}\n" + f"  Risk Constraint:  {'Enabled' if is_risk_constraint else 'Disabled'}\n" + "=" * 80 + "\n")
        start_time = time.time()
        # Call the fit_single_case function which returns fit results and fit time
        fit_revenue, fit_cvar, fit_status, fit_time_s, model = fit_single_case(n_scenarios, price_strategy.function, is_risk_constraint, verbose_market_data=True)
        end_time = time.time()
        # Report Fit results
        logger.info(f"\n--- Fit Results ---\nStatus:           {fit_status}\nFit Time:         {f'{fit_time_s:.2f} seconds' if fit_time_s is not None else 'N/A'}\nTotal Run Time:   {end_time - start_time:.2f} seconds\nFit Revenue:      {f'${fit_revenue:.2f}' if fit_revenue is not None else 'N/A'}\nFit CVaR:         {f'${fit_cvar:.2f}' if fit_cvar is not None else 'N/A'}")
        # --- Bids Reporting ---
        if model and model.all_bids:
             logger.info("\n--- Bids ---"); active_hours = []
             for hour_j in model.hours:
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
        elif model and fit_status not in ["optimal", "optimal_inaccurate"]: logger.warning(f"\nBids not generated due to fit status: {fit_status}")
        else: logger.warning(f"\nBids not generated (model fit failed or bids empty).")
        logger.info(f"\n{'=' * 80}\nReport saved to: {output_file}\n{'=' * 80}")
    except Exception as e: logger.exception(f"Error during detailed report generation: {e}")
    finally: logger.remove(console_id); logger.remove(file_id); logger.add(sys.stderr, level="INFO")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stochastic convergence bidding analysis")
    parser.add_argument("--analysis", action="store_true", help="Run multi-case analysis pipeline with OOS evaluation.")
    parser.add_argument("--scenarios", type=int, default=500, help="Number of scenarios for TRAIN data generation.") # Clarified help
    price_group = parser.add_mutually_exclusive_group()
    price_group.add_argument("--fixed-grid-resolution", type=float, metavar='RES', help="Fixed grid $ resolution.")
    price_group.add_argument("--num-price-points", type=int, metavar='N', help="N evenly spaced sample-based points.")
    price_group.add_argument("--use-all-unique", action="store_true", help="Use all unique prices from samples (default if no other).")
    price_group.add_argument("--every-nth", type=int, metavar='N', help="Use every Nth unique price from DA samples.")
    parser.add_argument("--round-decimals", type=int, help="Round prices to N decimal places before finding unique values. Use with --every-nth.")
    parser.add_argument("--no-risk-constraint", action="store_true", help="Disable the CVaR risk constraint.")
    parser.add_argument("--analysis-scenarios", type=str, default="500,1000,2000", help="CSV list of TRAIN scenario counts for analysis.") # Clarified help
    parser.add_argument("--analysis-resolutions", type=str, default="10,5,2,1", help="CSV list of fixed grid resolutions ($) for analysis.")
    parser.add_argument("--analysis-no-all-unique", action="store_true", help="Exclude 'All Unique Prices' from analysis.")
    # Optional: Add argument to override N_OOS_SCENARIOS
    parser.add_argument("--oos-scenarios", type=int, default=N_OOS_SCENARIOS, help=f"Number of scenarios for OOS evaluation dataset (default: {N_OOS_SCENARIOS}).")

    args = parser.parse_args()

    # Update N_OOS_SCENARIOS if provided via command line
    if args.oos_scenarios != N_OOS_SCENARIOS:
        if args.oos_scenarios <= 0:
             logger.error("--oos-scenarios must be positive.")
             sys.exit(1)
        N_OOS_SCENARIOS = args.oos_scenarios
        logger.info(f"Using {N_OOS_SCENARIOS} scenarios for OOS evaluation dataset.")


    # Set default price strategy if none specified
    if not args.analysis and not args.fixed_grid_resolution and not args.num_price_points and not args.every_nth: 
        args.use_all_unique = True

    risk_constraint = not args.no_risk_constraint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.analysis:
        logger.info(f"Mode: Running Analysis Pipeline with OOS Evaluation (N_OOS={N_OOS_SCENARIOS})")
        try: analysis_scenario_list = [int(s.strip()) for s in args.analysis_scenarios.split(',')]; assert analysis_scenario_list
        except: logger.error(f"Invalid --analysis-scenarios: '{args.analysis_scenarios}'"); sys.exit(1)
        strategy_configs = []
        if args.analysis_resolutions:
             try: resolutions = [float(r.strip()) for r in args.analysis_resolutions.split(',')]; assert all(res > 0 for res in resolutions)
             except: logger.error(f"Invalid --analysis-resolutions: '{args.analysis_resolutions}'"); sys.exit(1)
             for res in resolutions: strategy_configs.append({"strategy_type": "fixed_resolution", "fixed_resolution": res})
        else: logger.info("No fixed resolutions specified for analysis.")
        if not args.analysis_no_all_unique: strategy_configs.append({"strategy_type": "all_unique"}); logger.info("Including 'All Unique' price strategy.")
        else: logger.info("Excluding 'All Unique' price strategy.")
        if not strategy_configs: logger.error("No price strategies selected for analysis."); sys.exit(1)
        run_analysis_pipeline(scenario_list=analysis_scenario_list, price_strategy_configs=strategy_configs, use_risk_constraint=risk_constraint, timestamp=timestamp)
    else:
        logger.info("Mode: Running Single Detailed Report (Fit Only)")
        try:
            if args.fixed_grid_resolution: 
                assert args.fixed_grid_resolution > 0
                strategy = create_price_strategy(strategy_type="fixed_resolution", fixed_resolution=args.fixed_grid_resolution)
            elif args.num_price_points: 
                assert args.num_price_points > 0
                strategy = create_price_strategy(strategy_type="evenly_spaced", num_points=args.num_price_points)
            elif args.every_nth:
                assert args.every_nth > 0, "every-nth must be positive"
                # Create strategy with subsample_every_n type directly
                strategy = create_price_strategy(
                    strategy_type="subsample_every_n",
                    stride=args.every_nth,
                    round_decimals=args.round_decimals
                )
            elif args.use_all_unique: 
                strategy = create_price_strategy(strategy_type="all_unique")
            else: 
                logger.error("No price strategy specified for single run."); sys.exit(1)
                
            run_optimization_and_report_results(n_scenarios=args.scenarios, price_strategy=strategy, is_risk_constraint=risk_constraint)
        except (ValueError, AssertionError) as e: logger.error(f"Invalid argument for price strategy: {e}"); sys.exit(1)

# --- END OF REVISED run_project.py ---