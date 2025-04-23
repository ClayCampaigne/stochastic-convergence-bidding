# --- START OF UPDATED bidding_model.py ---

#!/usr/bin/env python3
"""
Bidding model class for stochastic convergence bidding optimization.
Implements the Sample-PV algorithm for convergence bidding.
Accepts a function to determine bid prices per hour.
"""

import time
from typing import Dict, List, Optional, Tuple, Callable # Added Callable

import cvxpy as cp
import numpy as np
import xarray as xr # Added xr
from numpy.typing import NDArray

# Assuming MarketData just holds the dataset now
from stochastic_convergence_bidding.market_data import MarketData

# Type alias for bid price functions (as defined in run_project.py)
BidPriceFunction = Callable[[xr.Dataset], np.ndarray]


class BiddingModel:
    """
    Builds and solves the sample-PV convergence bidding optimization
    for multiple hours with scenario-based CVaR, using a provided
    function to determine bid prices for each hour.
    """

    def __init__(
        self,
        market_data: MarketData,
        hours: List[int],
        get_bid_prices_f: BidPriceFunction, # Added: Function to get prices
        alpha: float = 0.95,
        rho: float = -1000.0,
        max_bid_volume_per_hour: float = 150.0,
        verbose: bool = False, # Defaulting verbose to False for solver
    ):
        """
        Initialize the bidding model with market data and parameters.

        Args:
            market_data: MarketData object containing dataset
            hours: List of hours to optimize for
            get_bid_prices_f: Function that takes hourly xr.Dataset and returns bid prices
            alpha: Quantile level for CVaR (default: 0.95)
            rho: CVaR threshold (expected shortfall level to control)
            max_bid_volume_per_hour: Maximum bid volume per hour
            verbose: Whether to print CVXPY solver output (default: False)
        """
        self.market_data = market_data
        self.hours = hours
        self.get_bid_prices_f = get_bid_prices_f # Store the function
        self.alpha = alpha
        self.rho = rho
        self.max_bid_volume_per_hour = max_bid_volume_per_hour
        self.verbose = verbose # Controls solver output

        self.sell_vars: List[cp.Variable] = []
        self.buy_vars: List[cp.Variable] = []
        self.bid_prices_per_hour: List[NDArray[np.float64]] = [] # Stores prices used

        self.problem: Optional[cp.Problem] = None
        self.objective_expr: Optional[cp.Expression] = None
        self.objective_value: Optional[float] = None

        # Final bid structures populated by postprocess_bids
        self.finalized_bids: dict[int, dict[str, dict[str, float]]] = {}
        self.all_bids: dict[int, dict[str, List[dict[str, float]]]] = {}

    def precompute_moneyness_bool_matrix(
        self, bid_prices_row_vector: np.ndarray, dalmp_column_vector: np.ndarray, is_sale: bool
    ) -> np.ndarray:
        """
        Create matrix of booleans, entry (i,j) is True if the DA bid price at column j is in the money for scenario i.
        (Unchanged from previous version)
        """
        if is_sale:
            is_in_the_money_matrix = bid_prices_row_vector <= dalmp_column_vector
        else:
            is_in_the_money_matrix = bid_prices_row_vector >= dalmp_column_vector
        return is_in_the_money_matrix

    def precompute_cleared_spread_matrix(
        self, hour: int, is_sale: bool, bid_prices: np.ndarray # Made bid_prices non-optional
    ) -> np.ndarray:
        """
        Compute matrix: each row i corresponds to a scenario, each column j corresponds to a candidate DA bid price p_j.
        Uses the provided bid_prices array.

        Args:
            hour: Hour to compute the spread matrix for
            is_sale: Flag indicating if this is for sell bids (True) or buy bids (False)
            bid_prices: Array of bid prices to use for this hour

        Returns:
            Matrix where each element (i,j) represents scenario revenue for scenario i at price j
        """
        # Removed the default fetching logic: bid_prices must be provided
        bid_prices = np.array(bid_prices) # Ensure ndarray

        ds = self.market_data.dataset
        # Ensure data exists for the hour before selecting
        if hour not in ds['hour'].values:
             raise ValueError(f"Hour {hour} not found in market data dataset.")
        hour_ds = ds.sel(hour=hour)
        dalmp = hour_ds["dalmp"].values
        rtlmp = hour_ds["rtlmp"].values

        # Ensure correct shapes for broadcasting
        bid_prices_row_vector = bid_prices.reshape(1, -1) # Explicit reshape
        dalmp_column_vector = dalmp.reshape(-1, 1)
        rtlmp_column_vector = rtlmp.reshape(-1, 1)

        dart_spread_column_vector = dalmp_column_vector - rtlmp_column_vector

        moneyness_bool_matrix = self.precompute_moneyness_bool_matrix(
            bid_prices_row_vector, dalmp_column_vector, is_sale
        )

        scenario_revenue_matrix = dart_spread_column_vector * moneyness_bool_matrix
        return scenario_revenue_matrix

    # Removed print_bids_w_nonzero_volumes - this can be handled by run_project.py
    # using the data in self.all_bids after postprocessing.

    def build_model(self, risk_constraint: bool):
        """
        Builds the CVXPY variables and constraints for each hour,
        plus CVaR constraints, volume caps, objective, etc.
        Uses self.get_bid_prices_f to determine prices per hour.
        """
        ds = self.market_data.dataset
        n_scenarios_total = ds.dims['scenario'] # Get total number of scenarios
        constraints: List[cp.Constraint] = []
        scenario_profits = []
        total_sale_expr = 0
        total_buy_expr = 0

        self.bid_prices_per_hour = [] # Clear previous prices if rebuilding
        self.sell_vars = []
        self.buy_vars = []

        # 1) For each hour, get prices, build spread matrix, define variables
        for hour_i in self.hours:
            # Get hour-specific data slice for the price function
            try:
                hour_data = ds.sel(hour=hour_i)
            except KeyError:
                 raise ValueError(f"Hour {hour_i} not found in dataset coordinates during build_model.")

            # *** Use the provided function to get bid prices ***
            bid_prices = self.get_bid_prices_f(hour_data)
            if not isinstance(bid_prices, np.ndarray):
                 bid_prices = np.array(bid_prices) # Ensure numpy array
            if bid_prices.ndim == 0: # Handle scalar case (e.g., if only one price)
                bid_prices = bid_prices.reshape(1,)
            if bid_prices.size == 0:
                 # Handle case with no bid prices gracefully? Or raise error?
                 # For now, let's skip this hour if no prices are generated.
                 print(f"Warning: No bid prices generated for hour {hour_i}. Skipping.")
                 # Add placeholders or handle as needed if skipping isn't desired
                 # We need to ensure lists maintain correct length if not skipping.
                 # Let's assume for now valid prices are always returned.
                 continue # Or raise error? Raising might be safer.
                 # raise ValueError(f"No bid prices generated by get_bid_prices_f for hour {hour_i}")

            self.bid_prices_per_hour.append(bid_prices)
            n_bids = len(bid_prices)

            # Build the "scenario x price" matrix using these prices
            sale_matrix = self.precompute_cleared_spread_matrix(hour_i, is_sale=True, bid_prices=bid_prices)
            buy_matrix = self.precompute_cleared_spread_matrix(hour_i, is_sale=False, bid_prices=bid_prices)

            # Check matrix dimensions consistency
            n_scenarios_hour, n_bids_sale = sale_matrix.shape
            _, n_bids_buy = buy_matrix.shape
            if n_scenarios_hour != n_scenarios_total:
                raise ValueError(f"Scenario dimension mismatch for hour {hour_i}. Expected {n_scenarios_total}, got {n_scenarios_hour}")
            if n_bids_sale != n_bids or n_bids_buy != n_bids:
                 raise ValueError(f"Bid dimension mismatch for hour {hour_i}. Expected {n_bids}, got Sale:{n_bids_sale}, Buy:{n_bids_buy}")

            # Define CVXPY variables for this hour
            w_sell = cp.Variable(n_bids, nonneg=True, name=f"w_sell_h{hour_i}")
            w_buy = cp.Variable(n_bids, nonneg=True, name=f"w_buy_h{hour_i}")
            self.sell_vars.append(w_sell)
            self.buy_vars.append(w_buy)

            # Hourly scenario revenues (vectorized)
            scenario_revenue_hour = (sale_matrix @ w_sell) - (buy_matrix @ w_buy)
            scenario_profits.append(scenario_revenue_hour)

            # Add to total volume expressions
            total_sale_expr += cp.sum(w_sell)
            total_buy_expr += cp.sum(w_buy)

        # Check if any hours were processed
        if not self.bid_prices_per_hour:
             raise ValueError("No hours were processed, possibly due to missing hours in data or errors generating bid prices.")

        # 2) Volume constraints (sum over hours <= total cap)
        total_cap = self.max_bid_volume_per_hour * len(self.hours) # Use intended hours length
        constraints.append(total_sale_expr <= total_cap)
        constraints.append(total_buy_expr <= total_cap)

        # 3) Combine scenario profits across hours
        # Ensure scenario_profits is not empty before summing
        if not scenario_profits:
             # This case should be caught earlier, but defensive check
             raise ValueError("scenario_profits list is empty, cannot build model.")

        total_scenario_profit = cp.sum(scenario_profits) # Sum list of CVXPY expressions

        # Check shape - should be (n_scenarios_total,)
        # CVXPY handles this implicitly, but good to keep in mind

        # 4) CVaR constraint
        t = cp.Variable(name="t_cvar")
        z = cp.Variable(n_scenarios_total, nonneg=True, name="z_cvar")

        if risk_constraint:
            # Ensure alpha is valid
            if not 0 < self.alpha < 1:
                 raise ValueError(f"CVaR alpha must be between 0 and 1, got {self.alpha}")
            constraints.append(z >= t - total_scenario_profit)
            constraints.append(t - (1.0 / ((1 - self.alpha) * n_scenarios_total)) * cp.sum(z) >= self.rho)

        # 5) Objective: maximize average profit
        if n_scenarios_total == 0:
             raise ValueError("Cannot compute average profit with zero scenarios.")
        sample_average_profit = cp.sum(total_scenario_profit) / n_scenarios_total
        objective = cp.Maximize(sample_average_profit)

        # 6) Create the CP problem
        self.problem = cp.Problem(objective, constraints)
        self.objective_expr = sample_average_profit # Store expression if needed later

    def solve_model(self, solver=cp.CLARABEL):
        """
        Solve the built model and store results internally.
        (Unchanged)
        """
        if self.problem is None:
            raise ValueError("Model not built. Call build_model() first.")

        start_time = time.time()
        # Capture solver output if self.verbose is True
        try:
            result = self.problem.solve(solver=solver, verbose=self.verbose)
            elapsed = time.time() - start_time
            self.objective_value = result # Can be None if solver fails

            if self.verbose:
                print(f"Solve completed in {elapsed:.2f} seconds. Status: {self.problem.status}")
            if self.problem.status not in ["optimal", "optimal_inaccurate"]:
                 print(f"Warning: Solver finished with status: {self.problem.status}")

        except Exception as e:
             print(f"Error during CVXPY solve: {e}")
             self.objective_value = None # Ensure objective is None on solver error
             # Potentially re-raise or handle more gracefully depending on needs
             raise # Re-raise the exception for now

        return self.objective_value # Return the objective value


    def get_solution(self) -> Tuple[
        List[NDArray[np.float64]],
        List[NDArray[np.float64]],
        Optional[float], # Objective can be None if solve fails
        List[NDArray[np.float64]],
    ]:
        """
        Return a list of sell decisions, buy decisions for each hour,
        the objective value, and the list of bid prices used for each hour.
        """
        if self.problem is None:
            # Check if it was built but solve failed
            if not self.bid_prices_per_hour:
                 raise ValueError("Model not built. Call build_model() first.")
            else:
                 print("Warning: Model built but problem/solution not available (likely solver failed). Returning empty decisions.")
                 # Return empty arrays matching the structure but with objective None
                 sell_decisions = [np.zeros(prices.shape) for prices in self.bid_prices_per_hour]
                 buy_decisions = [np.zeros(prices.shape) for prices in self.bid_prices_per_hour]
                 return sell_decisions, buy_decisions, self.objective_value, self.bid_prices_per_hour


        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem not solved optimally (Status: {self.problem.status}). Decision variables might be None.")

        sell_decisions = []
        buy_decisions = []

        if len(self.sell_vars) != len(self.bid_prices_per_hour) or len(self.buy_vars) != len(self.bid_prices_per_hour):
             raise ValueError("Mismatch between number of variables and number of price lists stored.")

        for i, bid_prices in enumerate(self.bid_prices_per_hour):
            w_s = self.sell_vars[i]
            w_b = self.buy_vars[i]
            # Handle case where variables might be None if solver failed badly
            sell_val = np.array(w_s.value) if w_s.value is not None else np.zeros_like(bid_prices)
            buy_val = np.array(w_b.value) if w_b.value is not None else np.zeros_like(bid_prices)
            sell_decisions.append(sell_val)
            buy_decisions.append(buy_val)

        # Return objective value (could be None)
        obj_value = self.objective_value

        return sell_decisions, buy_decisions, obj_value, self.bid_prices_per_hour

    def self_schedule_lower_bound(self):
        """ Compute lower bound via self-scheduling (assuming MarketData provides dataset). """
        # This method doesn't depend on get_bid_prices_f, likely OK as is.
        # Double check variable names match dataset ('dalmp', 'rtlmp')
        dataset = self.market_data.dataset
        hours = self.hours
        total_bid_volume_cap = self.max_bid_volume_per_hour * len(hours)

        n_hours = len(hours)
        n_scenarios = dataset.dims.get('scenario', 0)
        if n_scenarios == 0: raise ValueError("No scenarios in dataset for lower bound.")

        purchase_decisions = cp.Variable(n_hours, nonneg=True)
        sale_decisions = cp.Variable(n_hours, nonneg=True)

        # Ensure we select the correct hours
        ds_subset = dataset.sel(hour=hours)
        average_dalmp_by_hour = ds_subset["dalmp"].mean(dim="scenario").values
        average_rtlmp_by_hour = ds_subset["rtlmp"].mean(dim="scenario").values

        average_dart_spread_by_hour = average_dalmp_by_hour - average_rtlmp_by_hour

        constraints = [
            cp.sum(purchase_decisions) <= total_bid_volume_cap,
            cp.sum(sale_decisions) <= total_bid_volume_cap,
        ]

        # Use matrix multiplication for clarity
        expected_revenue_per_unit = average_dart_spread_by_hour
        total_expected_revenue = expected_revenue_per_unit @ sale_decisions - expected_revenue_per_unit @ purchase_decisions

        # Objective: Maximize average daily profit (original formula was per scenario?)
        # The objective is maximizing total expected profit across hours. Division by n_scenarios isn't needed here.
        objective = cp.Maximize(total_expected_revenue)

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != "optimal":
             print(f"Warning: Self-schedule lower bound solve status: {problem.status}")
             # Return NaNs or handle appropriately
             return np.nan, np.full(n_hours, np.nan), np.full(n_hours, np.nan)

        return problem.value, purchase_decisions.value, sale_decisions.value


    def perfect_foresight_upper_bound(self):
        """ Compute upper bound via perfect foresight (assuming MarketData provides dataset). """
        # This method also doesn't depend on get_bid_prices_f, likely OK.
        dataset = self.market_data.dataset
        hours = self.hours
        max_bid_volume_per_hour = self.max_bid_volume_per_hour
        total_volume_cap = max_bid_volume_per_hour * len(hours)

        n_scenarios = dataset.dims.get('scenario', 0)
        if n_scenarios == 0: raise ValueError("No scenarios in dataset for upper bound.")

        obj_vals = []
        for scenario_idx in range(n_scenarios):
            # Select data for the specific scenario and hours
            scenario_ds = dataset.sel(scenario=scenario_idx, hour=hours)
            dalmps = scenario_ds["dalmp"].values
            rtlmps = scenario_ds["rtlmp"].values

            dart_spread = dalmps - rtlmps # DART spread for each hour in this scenario

            sale_decisions = cp.Variable(len(hours), nonneg=True)
            purchase_decisions = cp.Variable(len(hours), nonneg=True)
            constraints = [
                cp.sum(sale_decisions) <= total_volume_cap,
                cp.sum(purchase_decisions) <= total_volume_cap,
            ]

            # Maximize profit for this specific scenario
            revenue = dart_spread @ sale_decisions - dart_spread @ purchase_decisions

            objective = cp.Maximize(revenue)
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status != "optimal":
                # Handle non-optimal solve for a scenario
                print(f"Warning: Perfect foresight upper bound solve status for scenario {scenario_idx}: {problem.status}")
                obj_vals.append(np.nan) # Append NaN or 0? NaN seems more appropriate
            else:
                 obj_vals.append(problem.value)

        # Return the average over scenarios, ignoring NaNs
        return np.nanmean(obj_vals)


    def postprocess_bids(self) -> None:
        """
        After solve_model() is done, process the raw solution vectors into
        bid/offer information. Uses results from get_solution().
        """
        # Get the solution vectors - handles None objective value
        sell_decision_vectors, buy_decision_vectors, obj_val, bid_price_vectors = self.get_solution()

        self.finalized_bids = {}
        self.all_bids = {}

        # Ensure we iterate using the correct list of hours used in the model
        processed_hours = self.hours # Assume self.hours matches the processed hours

        if len(sell_decision_vectors) != len(processed_hours) or \
           len(buy_decision_vectors) != len(processed_hours) or \
           len(bid_price_vectors) != len(processed_hours):
            # This might happen if build_model skipped hours or get_solution had issues
            print(f"Warning: Mismatch in lengths during postprocessing. Processed Hours: {len(processed_hours)}, Sell Vecs: {len(sell_decision_vectors)}, Buy Vecs: {len(buy_decision_vectors)}, Price Vecs: {len(bid_price_vectors)}")
            # Decide how to handle: maybe only process matching indices? For now, proceed cautiously.
            # Let's assume lengths match based on current structure.

        # Iterate over each hour index corresponding to the model's processed hours
        for i, hour_i in enumerate(processed_hours):
            # Check if index exists in the result lists (safety for mismatch)
            if i >= len(sell_decision_vectors): break

            w_sell = np.round(sell_decision_vectors[i], 3) # Use more precision maybe?
            w_buy = np.round(buy_decision_vectors[i], 3)
            prices = bid_price_vectors[i]

            # Ensure prices is not empty
            if prices.size == 0:
                print(f"Warning: Empty price vector for hour {hour_i} during postprocessing.")
                self.all_bids[hour_i] = {"sell": [], "buy": []}
                self.finalized_bids[hour_i] = {"sell": {"price": np.nan, "volume_mw": 0.0}, "buy": {"price": np.nan, "volume_mw": 0.0}}
                continue

            # --- Store ALL non-zero bids ---
            hour_all_bids: Dict[str, List[Dict[str, float]]] = {"sell": [], "buy": []}

            # Add all non-zero sell bids
            for price, volume in zip(prices, w_sell):
                if volume > 1e-3: # Use tolerance instead of exact zero
                    hour_all_bids["sell"].append({"price": float(price), "volume_mw": float(volume)})

            # Add all non-zero buy bids
            for price, volume in zip(prices, w_buy):
                if volume > 1e-3: # Use tolerance
                    hour_all_bids["buy"].append({"price": float(price), "volume_mw": float(volume)})

            # Sort bids by price (conventional market representation)
            # Sell offers: lowest price first
            # Buy bids: highest price first
            hour_all_bids["sell"].sort(key=lambda x: x["price"])
            hour_all_bids["buy"].sort(key=lambda x: x["price"], reverse=True)

            self.all_bids[hour_i] = hour_all_bids

            # --- Store largest-volume bids for backward compatibility (Optional) ---
            # This part might be less meaningful if bids are spread across many prices
            best_sell_price, best_sell_vol = (max(hour_all_bids["sell"], key=lambda x: x["volume_mw"]).values()
                                              if hour_all_bids["sell"] else (np.nan, 0.0))
            best_buy_price, best_buy_vol = (max(hour_all_bids["buy"], key=lambda x: x["volume_mw"]).values()
                                             if hour_all_bids["buy"] else (np.nan, 0.0))

            self.finalized_bids[hour_i] = {
                "sell": {"price": float(best_sell_price), "volume_mw": float(best_sell_vol)},
                "buy": {"price": float(best_buy_price), "volume_mw": float(best_buy_vol)},
            }


# --- END OF UPDATED bidding_model.py ---