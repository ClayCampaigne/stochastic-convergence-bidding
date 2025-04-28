# --- START OF UPDATED bidding_model.py with fit/evaluate_oos ---

#!/usr/bin/env python3
"""
Bidding model class for stochastic convergence bidding optimization.
Implements the Sample-PV algorithm for convergence bidding.
Accepts a function to determine bid prices per hour.
Includes methods for fitting the model and evaluating out-of-sample performance.
"""

import time
from typing import Dict, List, Optional, Tuple, Callable, Union # Added Union

import cvxpy as cp
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from loguru import logger # Using logger for warnings/errors

# Assuming MarketData just holds the dataset now
from stochastic_convergence_bidding.market_data import MarketData

# Type alias for bid price functions (as defined in run_project.py)
BidPriceFunction = Callable[[xr.Dataset], np.ndarray]


# Helper function to calculate CVaR
def calculate_empirical_cvar(profits: np.ndarray, alpha: float) -> Optional[float]:
    """Calculates the empirical CVaR for a given profit distribution."""
    if not isinstance(profits, np.ndarray) or profits.ndim != 1 or profits.size == 0:
        logger.error("Invalid input for CVaR calculation: Expected 1D numpy array.")
        return None
    if not 0 < alpha < 1:
        logger.error(f"Invalid alpha for CVaR calculation: {alpha}. Must be (0, 1).")
        return None

    n_scenarios = len(profits)
    q = 100 * (1 - alpha) # Percentile for VaR (lower tail for profits)

    # Handle potential NaNs
    valid_profits = profits[~np.isnan(profits)]
    if valid_profits.size == 0:
        logger.warning("No valid profit scenarios for CVaR calculation.")
        return None

    if valid_profits.size < 1 / (1 - alpha):
         logger.warning(f"Number of scenarios ({valid_profits.size}) is small relative to alpha ({alpha}), CVaR estimate may be unreliable.")

    var = np.percentile(valid_profits, q)
    cvar = np.mean(valid_profits[valid_profits <= var])

    return cvar


class BiddingModel:
    """
    Builds, solves (fits), and evaluates the sample-PV convergence
    bidding optimization for multiple hours with scenario-based CVaR.

    Requires a function to determine bid prices for each hour.
    Out-of-sample evaluation assumes the SAME bid prices are used
    (i.e., requires a data-independent price strategy like fixed_grid_prices).
    """

    def __init__(
        self,
        train_market_data: MarketData, # Renamed for clarity
        hours: List[int],
        get_bid_prices_f: BidPriceFunction,
        alpha: float = 0.95,
        rho: float = -1000.0,
        max_bid_volume_per_hour: float = 150.0,
        verbose_solver: bool = False, # Renamed for clarity
    ):
        """
        Initialize the bidding model.

        Args:
            train_market_data: MarketData object containing training dataset
            hours: List of hours to optimize for
            get_bid_prices_f: Function that takes hourly xr.Dataset and returns bid prices.
                              IMPORTANT: For evaluate_oos to work correctly, this function
                              MUST return the same prices regardless of the input dataset
                              (e.g., created by fixed_grid_prices).
            alpha: Quantile level for CVaR (default: 0.95)
            rho: CVaR threshold (expected shortfall level to control)
            max_bid_volume_per_hour: Maximum bid volume per hour
            verbose_solver: Whether to print CVXPY solver output (default: False)
        """
        self.train_market_data = train_market_data
        self.hours = hours
        self.get_bid_prices_f = get_bid_prices_f
        self.alpha = alpha
        self.rho = rho
        self.max_bid_volume_per_hour = max_bid_volume_per_hour
        self.verbose_solver = verbose_solver

        # --- Model Internals (populated by _build_model) ---
        self.sell_vars: List[cp.Variable] = []
        self.buy_vars: List[cp.Variable] = []
        self.problem: Optional[cp.Problem] = None
        self.objective_expr: Optional[cp.Expression] = None
        self.cvar_t: Optional[cp.Variable] = None
        self.cvar_z: Optional[cp.Variable] = None

        # --- Results from fit() ---
        self.bid_prices_per_hour: List[NDArray[np.float64]] = [] # Prices used during fit
        self.trained_sell_volumes: Optional[List[NDArray[np.float64]]] = None
        self.trained_buy_volumes: Optional[List[NDArray[np.float64]]] = None
        self.fit_objective_value: Optional[float] = None
        self.fit_status: Optional[str] = None
        self.fit_cvar_value: Optional[float] = None # Store IS CVaR if needed

        # --- Bids (populated by _postprocess_bids after fit) ---
        self.all_bids: dict[int, dict[str, List[dict[str, float]]]] = {}

    # --- Private Helper for Calculating Spread Matrix ---

    def _calculate_spread_matrix(
        self, market_data: MarketData, hour: int, is_sale: bool, bid_prices: np.ndarray
    ) -> np.ndarray:
        """ Calculates the scenario_revenue_matrix for a given dataset and prices. """
        bid_prices = np.asarray(bid_prices) # Ensure ndarray

        ds = market_data.dataset
        if hour not in ds['hour'].values:
             raise ValueError(f"Hour {hour} not found in market data dataset.")
        hour_ds = ds.sel(hour=hour)
        dalmp = hour_ds["dalmp"].values
        rtlmp = hour_ds["rtlmp"].values

        if dalmp.ndim == 0: dalmp = dalmp.reshape(1,) # Handle single scenario data
        if rtlmp.ndim == 0: rtlmp = rtlmp.reshape(1,)

        bid_prices_row_vector = bid_prices.reshape(1, -1)
        dalmp_column_vector = dalmp.reshape(-1, 1)
        rtlmp_column_vector = rtlmp.reshape(-1, 1)

        dart_spread_column_vector = dalmp_column_vector - rtlmp_column_vector

        moneyness_bool_matrix = self._precompute_moneyness_bool_matrix(
            bid_prices_row_vector, dalmp_column_vector, is_sale
        )

        scenario_revenue_matrix = dart_spread_column_vector * moneyness_bool_matrix
        return scenario_revenue_matrix

    # --- Internal Model Building Logic ---

    def _precompute_moneyness_bool_matrix( # Renamed with underscore
        self, bid_prices_row_vector: np.ndarray, dalmp_column_vector: np.ndarray, is_sale: bool
    ) -> np.ndarray:
        """ Creates matrix of booleans indicating if a bid price is in the money. """
        if is_sale:
            is_in_the_money_matrix = bid_prices_row_vector <= dalmp_column_vector
        else:
            is_in_the_money_matrix = bid_prices_row_vector >= dalmp_column_vector
        return is_in_the_money_matrix

    def _build_model(self, risk_constraint: bool):
        """
        Builds the CVXPY optimization problem using the training data.
        Populates self.problem, self.sell_vars, self.buy_vars, etc.
        """
        logger.debug("Building optimization model...")
        ds = self.train_market_data.dataset # Use training data
        n_scenarios_total = ds.sizes.get('scenario', 0)
        if n_scenarios_total == 0: raise ValueError("Training data has zero scenarios.")

        constraints: List[cp.Constraint] = []
        scenario_profits = []
        total_sale_expr = 0
        total_buy_expr = 0

        # Clear previous build results
        self.bid_prices_per_hour = []
        self.sell_vars = []
        self.buy_vars = []
        self.problem = None
        self.objective_expr = None
        self.cvar_t = None
        self.cvar_z = None

        # 1) Process each hour
        for hour_i in self.hours:
            try: hour_data = ds.sel(hour=hour_i)
            except KeyError: raise ValueError(f"Hour {hour_i} not found in training dataset.")

            bid_prices = self.get_bid_prices_f(hour_data) # Use stored function
            bid_prices = np.asarray(bid_prices)
            if bid_prices.ndim == 0: bid_prices = bid_prices.reshape(1,)
            if bid_prices.size == 0:
                 logger.error(f"No bid prices generated by get_bid_prices_f for hour {hour_i}. Cannot build model.")
                 raise ValueError(f"No bid prices generated for hour {hour_i}")

            self.bid_prices_per_hour.append(bid_prices)
            n_bids = len(bid_prices)

            # Calculate spread matrices using *training* data
            sale_matrix = self._calculate_spread_matrix(self.train_market_data, hour_i, is_sale=True, bid_prices=bid_prices)
            buy_matrix = self._calculate_spread_matrix(self.train_market_data, hour_i, is_sale=False, bid_prices=bid_prices)

            # Dimension checks
            n_scen_h, n_bids_s = sale_matrix.shape; _, n_bids_b = buy_matrix.shape
            if n_scen_h != n_scenarios_total: raise ValueError(f"Scenario dim mismatch hour {hour_i}")
            if n_bids_s != n_bids or n_bids_b != n_bids: raise ValueError(f"Bid dim mismatch hour {hour_i}")

            # Define variables
            w_sell = cp.Variable(n_bids, nonneg=True, name=f"w_sell_h{hour_i}")
            w_buy = cp.Variable(n_bids, nonneg=True, name=f"w_buy_h{hour_i}")
            self.sell_vars.append(w_sell); self.buy_vars.append(w_buy)

            # Hourly scenario revenues expression
            scenario_revenue_hour = (sale_matrix @ w_sell) - (buy_matrix @ w_buy)
            scenario_profits.append(scenario_revenue_hour)

            # Total volume expressions
            total_sale_expr += cp.sum(w_sell); total_buy_expr += cp.sum(w_buy)

        if not self.bid_prices_per_hour:
             raise ValueError("Model building failed: No hours processed.")

        # 2) Volume constraints
        total_cap = self.max_bid_volume_per_hour * len(self.hours)
        constraints.append(total_sale_expr <= total_cap)
        constraints.append(total_buy_expr <= total_cap)

        # 3) Combine profits
        if not scenario_profits: raise ValueError("scenario_profits list is empty.")
        total_scenario_profit = cp.sum(scenario_profits) # CVXPY handles summing expressions

        # 4) CVaR constraint
        if risk_constraint:
            if not 0 < self.alpha < 1: raise ValueError(f"CVaR alpha must be between 0 and 1, got {self.alpha}")
            t = cp.Variable(name="t_cvar")
            z = cp.Variable(n_scenarios_total, nonneg=True, name="z_cvar")
            constraints.append(z >= t - total_scenario_profit)
            # Check division by zero if alpha is near 1 or n_scenarios_total is 0 (already checked)
            cvar_denominator = (1 - self.alpha) * n_scenarios_total
            if abs(cvar_denominator) < 1e-9: raise ValueError("CVaR denominator too small (alpha near 1 or zero scenarios).")
            constraints.append(t - (1.0 / cvar_denominator) * cp.sum(z) >= self.rho)
            
            # Store CVaR variables for potential post-solve calculations
            self.cvar_t = t
            self.cvar_z = z

        # 5) Objective
        sample_average_profit = cp.sum(total_scenario_profit) / n_scenarios_total
        objective = cp.Maximize(sample_average_profit)

        # 6) Create problem object
        self.problem = cp.Problem(objective, constraints)
        self.objective_expr = sample_average_profit
        logger.debug("Optimization model built successfully.")


    # --- Internal Model Solving Logic ---

    def _solve_model(self, solver=cp.CLARABEL) -> Tuple[Optional[float], str]:
        """ Solves the CVXPY problem stored in self.problem. """
        if self.problem is None:
            logger.error("Cannot solve model: Problem not built.")
            return None, "error_not_built"

        logger.debug(f"Solving model using {solver}...")
        start_time = time.time()
        objective_value = None
        status = "error_solver"
        try:
            # Use self.verbose_solver to control CVXPY output
            objective_value = self.problem.solve(solver=solver, verbose=self.verbose_solver)
            elapsed = time.time() - start_time
            status = self.problem.status
            logger.debug(f"Solve completed in {elapsed:.2f}s. Status: {status}")
            if status not in ["optimal", "optimal_inaccurate"]:
                 logger.warning(f"Solver finished with non-optimal status: {status}")

        except Exception as e:
             logger.exception(f"Error during CVXPY solve: {e}")
             # objective_value remains None

        return objective_value, status

    # --- Internal Bid Postprocessing ---

    def _postprocess_bids(self) -> None:
        """ Populates self.all_bids based on solved variable values. """
        if self.fit_status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Cannot postprocess bids: Model fit status was '{self.fit_status}'.")
            self.all_bids = {}
            return

        if self.trained_sell_volumes is None or self.trained_buy_volumes is None:
             logger.error("Cannot postprocess bids: Trained volumes not available.")
             self.all_bids = {}
             return

        self.all_bids = {}
        processed_hours = self.hours # Assumes these match the fitted hours

        if len(self.trained_sell_volumes) != len(processed_hours) or \
           len(self.trained_buy_volumes) != len(processed_hours) or \
           len(self.bid_prices_per_hour) != len(processed_hours):
            logger.error("Mismatch in lengths during postprocessing. Cannot generate bids.")
            return

        for i, hour_i in enumerate(processed_hours):
            w_sell = np.round(self.trained_sell_volumes[i], 3)
            w_buy = np.round(self.trained_buy_volumes[i], 3)
            prices = self.bid_prices_per_hour[i]

            if prices.size == 0:
                logger.warning(f"Empty price vector for hour {hour_i} during postprocessing.")
                self.all_bids[hour_i] = {"sell": [], "buy": []}
                continue

            hour_all_bids: Dict[str, List[Dict[str, float]]] = {"sell": [], "buy": []}
            for price, volume in zip(prices, w_sell):
                if volume > 1e-3: hour_all_bids["sell"].append({"price": float(price), "volume_mw": float(volume)})
            for price, volume in zip(prices, w_buy):
                if volume > 1e-3: hour_all_bids["buy"].append({"price": float(price), "volume_mw": float(volume)})

            hour_all_bids["sell"].sort(key=lambda x: x["price"])
            hour_all_bids["buy"].sort(key=lambda x: x["price"], reverse=True)
            self.all_bids[hour_i] = hour_all_bids
        logger.debug("Bids postprocessed successfully.")
        
    def _calculate_scenario_profits(self, market_data: MarketData) -> Optional[np.ndarray]:
        """Helper to calculate scenario profits for a given dataset using stored trained volumes."""
        if self.trained_sell_volumes is None or self.trained_buy_volumes is None:
            logger.error("Cannot calculate profits: Trained volumes not available.")
            return None
        if len(self.trained_sell_volumes) != len(self.hours) or \
           len(self.trained_buy_volumes) != len(self.hours) or \
           len(self.bid_prices_per_hour) != len(self.hours):
            logger.error("Mismatch in lengths for profit calculation.")
            return None

        scenario_profits_list = []
        try:
            for i, hour_i in enumerate(self.hours):
                bid_prices = self.bid_prices_per_hour[i]
                w_sell_trained = self.trained_sell_volumes[i]
                w_buy_trained = self.trained_buy_volumes[i]

                sale_matrix = self._calculate_spread_matrix(market_data, hour_i, is_sale=True, bid_prices=bid_prices)
                buy_matrix = self._calculate_spread_matrix(market_data, hour_i, is_sale=False, bid_prices=bid_prices)

                revenue_hour = (sale_matrix @ w_sell_trained) - (buy_matrix @ w_buy_trained)
                scenario_profits_list.append(revenue_hour)

            if not scenario_profits_list: return None
            total_scenario_profit = np.sum(np.array(scenario_profits_list), axis=0)
            return total_scenario_profit

        except Exception as e:
            logger.exception(f"Error calculating scenario profits: {e}")
            return None

    # --- Public Methods: Fit and Evaluate ---

    def fit(self, risk_constraint: bool = True, solver=cp.CLARABEL) -> None:
        """
        Builds and solves the optimization model using the training data.
        Stores the trained volumes and objective value.

        Args:
            risk_constraint: Whether to include the CVaR constraint.
            solver: The CVXPY solver to use.
        """
        logger.info("Fitting the bidding model...")
        self.fit_cvar_value = None # Reset CVaR value
        try:
            # Build the optimization problem
            self._build_model(risk_constraint=risk_constraint)

            # Solve the problem
            objective_value, status = self._solve_model(solver=solver)
            self.fit_objective_value = objective_value
            self.fit_status = status

            # Store results if solve was successful
            if status in ["optimal", "optimal_inaccurate"]:
                if len(self.sell_vars) != len(self.bid_prices_per_hour) or \
                   len(self.buy_vars) != len(self.bid_prices_per_hour):
                    logger.error("Mismatch between CVXPY variables and stored bid prices after solve.")
                    self.trained_sell_volumes = None
                    self.trained_buy_volumes = None
                else:
                    # Extract solution values
                    self.trained_sell_volumes = []
                    self.trained_buy_volumes = []
                    for i, prices in enumerate(self.bid_prices_per_hour):
                         sell_val = self.sell_vars[i].value
                         buy_val = self.buy_vars[i].value
                         # Handle None values defensively, though optimal status should prevent this
                         self.trained_sell_volumes.append(np.array(sell_val) if sell_val is not None else np.zeros_like(prices))
                         self.trained_buy_volumes.append(np.array(buy_val) if buy_val is not None else np.zeros_like(prices))

                    # Populate the self.all_bids structure
                    self._postprocess_bids()
                    
                    # Calculate and store In-Sample CVaR value
                    if risk_constraint and self.cvar_t is not None and self.cvar_z is not None:
                        try:
                            train_profits = self._calculate_scenario_profits(self.train_market_data)
                            if train_profits is not None:
                                self.fit_cvar_value = calculate_empirical_cvar(train_profits, self.alpha)
                                logger.info(f"Model fit complete. Status: {status}, Objective: {objective_value:.4f}, CVaR({self.alpha:.2f}): {self.fit_cvar_value:.4f}")
                            else:
                                logger.warning("Could not calculate IS CVaR after fit (profit calc failed).")
                                logger.info(f"Model fit complete. Status: {status}, Objective: {objective_value:.4f}")
                        except Exception as cvar_err:
                            logger.warning(f"Could not calculate IS CVaR after fit: {cvar_err}")
                            logger.info(f"Model fit complete. Status: {status}, Objective: {objective_value:.4f}")
                    else:
                        logger.info(f"Model fit complete. Status: {status}, Objective: {objective_value:.4f}")

            else:
                # Solve failed or was non-optimal
                self.trained_sell_volumes = None
                self.trained_buy_volumes = None
                self.all_bids = {} # Ensure bids are empty
                logger.error(f"Model fit failed or non-optimal. Status: {status}. Trained volumes not stored.")

        except Exception as e:
            logger.exception("Error during model fitting process.")
            self.fit_status = "error_fit_exception"
            self.fit_objective_value = None
            self.trained_sell_volumes = None
            self.trained_buy_volumes = None
            self.all_bids = {}


    def evaluate_oos(self, test_market_data: MarketData) -> Tuple[Optional[float], Optional[float], str]:
        """
        Evaluates the fitted model on out-of-sample (test) data.

        Requires that the model has been successfully fitted (`fit()` called)
        and that the `get_bid_prices_f` used during initialization produces
        the *exact same* price points for the test data as it did for the
        training data (typically requiring a fixed price grid strategy).

        Args:
            test_market_data: MarketData object containing the test dataset.

        Returns:
            Tuple: (Average OOS Profit, OOS CVaR, Status String)
                   Status can be 'success', 'fit_not_optimal', 'not_fitted',
                   'data_error', 'price_mismatch', 'calculation_error'.
        """
        logger.info("Evaluating model out-of-sample...")
        status = "evaluation_started"

        # 1. Check if model was fitted
        if self.trained_sell_volumes is None or self.trained_buy_volumes is None:
            logger.error("Cannot evaluate OOS: Model has not been successfully fitted yet (call fit()).")
            status = "not_fitted"
            return None, None, status
        if not self.bid_prices_per_hour:
            logger.error("Cannot evaluate OOS: No bid prices were stored during fit.")
            status = "not_fitted"
            return None, None, status
        # Check fit status again, although volumes check should cover this
        if self.fit_status not in ["optimal", "optimal_inaccurate"]:
            logger.warning(f"Cannot evaluate OOS: Model fit status was '{self.fit_status}'.")
            status = "fit_not_optimal"
            return None, None, status

        test_ds = test_market_data.dataset
        n_scenarios_test = test_ds.dims.get('scenario', 0)
        if n_scenarios_test == 0:
            logger.error("Cannot evaluate OOS: Test data has zero scenarios.")
            status = "data_error"
            return None, None, status

        # Calculate total scenario profits on OOS data
        total_oos_scenario_profit = self._calculate_scenario_profits(test_market_data)
        
        if total_oos_scenario_profit is None:
            logger.error("Failed to calculate OOS scenario profits.")
            status = "calculation_error"
            return None, None, status

        # --- CRITICAL CHECK for price strategy consistency ---
        # This check is already handled in _calculate_scenario_profits,
        # but we'll do an explicit check here for clarity
        price_mismatch = False
        for i, hour_i in enumerate(self.hours):
            train_bid_prices = self.bid_prices_per_hour[i]
            try:
                test_hour_data = test_ds.sel(hour=hour_i)
                test_bid_prices = np.asarray(self.get_bid_prices_f(test_hour_data))
                if test_bid_prices.ndim == 0: test_bid_prices = test_bid_prices.reshape(1,)
                if not np.array_equal(train_bid_prices, test_bid_prices):
                    price_mismatch = True
                    logger.error(f"OOS Price Mismatch Hour {hour_i}. Train: {train_bid_prices[:3]}..., Test: {test_bid_prices[:3]}...")
                    break # Stop checking on first mismatch
            except Exception as e:
                logger.error(f"Error checking prices OOS hour {hour_i}: {e}")
                price_mismatch = True
                break # Treat error as mismatch
        if price_mismatch:
            logger.error("OOS evaluation failed due to price mismatch. Requires fixed price strategy.")
            status = "price_mismatch"
            return None, None, status
        # --- End Price Check ---

        # Calculate metrics
        avg_oos_profit = np.mean(total_oos_scenario_profit)
        cvar_oos = calculate_empirical_cvar(total_oos_scenario_profit, self.alpha)

        if cvar_oos is None:
            logger.warning("Could not calculate OOS CVaR.")
            status = "cvar_calculation_error"
        else:
            status = "success"

        logger.info(f"Out-of-sample evaluation complete. Average Profit: {avg_oos_profit:.4f}, OOS CVaR({self.alpha:.2f}): {cvar_oos if cvar_oos is not None else 'N/A'}")
        return avg_oos_profit, cvar_oos, status


    # --- Optional: Helper Methods (Lower/Upper Bounds) ---
    # These remain largely independent but use self.train_market_data now

    def self_schedule_lower_bound(self) -> Tuple[Optional[float], Optional[NDArray], Optional[NDArray]]:
        """ Compute lower bound via self-scheduling on training data. """
        logger.debug("Calculating self-schedule lower bound...")
        dataset = self.train_market_data.dataset
        hours = self.hours
        total_bid_volume_cap = self.max_bid_volume_per_hour * len(hours)
        n_hours = len(hours)
        n_scenarios = dataset.dims.get('scenario', 0)
        if n_scenarios == 0: logger.error("No scenarios in dataset for lower bound."); return None, None, None

        try:
            purchase_decisions = cp.Variable(n_hours, nonneg=True)
            sale_decisions = cp.Variable(n_hours, nonneg=True)
            ds_subset = dataset.sel(hour=hours)
            avg_dalmp = ds_subset["dalmp"].mean(dim="scenario").values
            avg_rtlmp = ds_subset["rtlmp"].mean(dim="scenario").values
            avg_dart = avg_dalmp - avg_rtlmp
            constraints = [cp.sum(purchase_decisions) <= total_bid_volume_cap, cp.sum(sale_decisions) <= total_bid_volume_cap]
            total_expected_revenue = avg_dart @ sale_decisions - avg_dart @ purchase_decisions
            # perturb costs slightly to check for degeneracy
            # penalized_revenue = total_expected_revenue - 0.01 * (cp.sum(sale_decisions) + cp.sum(purchase_decisions))
            sale_penalty_coef = np.random.uniform(-0.0001, 0.0001, n_hours)
            purchase_penalty_coef = np.random.uniform(-0.0001, 0.0001, n_hours)
            penalized_revenue = total_expected_revenue - cp.sum(sale_penalty_coef * sale_decisions) - cp.sum(purchase_penalty_coef * purchase_decisions)
            objective = cp.Maximize(penalized_revenue)
            problem = cp.Problem(objective, constraints)
            problem.solve()

            if problem.status != "optimal":
                 logger.warning(f"Self-schedule lower bound solve status: {problem.status}")
                 return None, None, None
            return problem.value, purchase_decisions.value, sale_decisions.value
        except Exception as e:
            logger.exception("Error calculating self-schedule lower bound.")
            return None, None, None


    def perfect_foresight_upper_bound(self) -> Optional[float]:
        """ Compute upper bound via perfect foresight on training data. """
        logger.debug("Calculating perfect foresight upper bound...")
        dataset = self.train_market_data.dataset
        hours = self.hours
        total_volume_cap = self.max_bid_volume_per_hour * len(hours)
        n_scenarios = dataset.dims.get('scenario', 0)
        if n_scenarios == 0: logger.error("No scenarios in dataset for upper bound."); return None

        obj_vals = []
        try:
            for scenario_idx in range(n_scenarios):
                scenario_ds = dataset.sel(scenario=scenario_idx, hour=hours)
                dalmps = scenario_ds["dalmp"].values; rtlmps = scenario_ds["rtlmp"].values
                dart_spread = dalmps - rtlmps
                sale_dec = cp.Variable(len(hours), nonneg=True); purch_dec = cp.Variable(len(hours), nonneg=True)
                constraints = [cp.sum(sale_dec) <= total_volume_cap, cp.sum(purch_dec) <= total_volume_cap]
                revenue = dart_spread @ sale_dec - dart_spread @ purch_dec
                objective = cp.Maximize(revenue)
                problem = cp.Problem(objective, constraints)
                problem.solve()
                if problem.status != "optimal":
                    logger.warning(f"Perfect foresight upper bound solve status for scenario {scenario_idx}: {problem.status}")
                    obj_vals.append(np.nan)
                else: obj_vals.append(problem.value)
            return np.nanmean(obj_vals) # Average results
        except Exception as e:
            logger.exception("Error calculating perfect foresight upper bound.")
            return None


# --- END OF UPDATED bidding_model.py ---