#!/usr/bin/env python3
"""
Bidding model class for stochastic convergence bidding optimization.
Implements the Sample-PV algorithm for convergence bidding.
"""

import time
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from stochastic_convergence_bidding.market_data import MarketData


class BiddingModel:
    """
    Builds and solves the sample-PV convergence bidding optimization
    for multiple hours with scenario-based CVaR.
    """

    def __init__(
        self,
        market_data: MarketData,
        hours: List[int],
        alpha: float = 0.95,
        rho: float = -1000.0,
        max_bid_volume_per_hour: float = 150.0,
        verbose: bool = True,
    ):
        """
        Initialize the bidding model with market data and parameters.

        Args:
            market_data: MarketData object containing dataset and utility methods
            hours: List of hours to optimize for
            alpha: Quantile level for CVaR (default: 0.95)
            rho: CVaR threshold (expected shortfall level to control)
            max_bid_volume_per_hour: Maximum bid volume per hour
            verbose: Whether to print solver output (default: True)
        """
        self.market_data = market_data
        self.hours = hours
        self.alpha = alpha
        self.rho = rho
        self.max_bid_volume_per_hour = max_bid_volume_per_hour
        self.verbose = verbose

        self.sell_vars: List[cp.Variable] = []  # list of cp.Variables
        self.buy_vars: List[cp.Variable] = []  # list of cp.Variables
        self.bid_prices_per_hour: List[NDArray[np.float64]] = []  # list of np.ndarray

        # We'll build the problem in build_model()
        self.problem: Optional[cp.Problem] = None
        self.objective_expr: Optional[cp.Expression] = None

        # We'll store the final solution or objective after solve().
        self.objective_value: Optional[float] = None
        
        # After solving and postprocessing, we can store the final bids here:
        self.finalized_bids: dict[int, dict[str, dict[str, float]]] = {}  # dict: hour -> { "buy": {"price": float, "volume_mw": float}, "sell": {"price": float, "volume_mw": float} }
        
        # New attribute to store all the bids, not just the largest one
        self.all_bids: dict[int, dict[str, List[dict[str, float]]]] = {}  # dict: hour -> { "buy": [{"price": float, "volume_mw": float}, ...], "sell": [{"price": float, "volume_mw": float}, ...] }

    def precompute_moneyness_bool_matrix(
        self, bid_prices_row_vector: np.ndarray, dalmp_column_vector: np.ndarray, is_sale: bool
    ) -> np.ndarray:
        """
        Create matrix of booleans, entry (i,j) is True if the DA bid price at column j is in the money for scenario j.
        
        Args:
            bid_prices_row_vector: Row vector of bid prices
            dalmp_column_vector: Column vector of DA prices
            is_sale: Flag indicating if this is for sell bids (True) or buy bids (False)
            
        Returns:
            Matrix of booleans
        """
        if is_sale:
            # True if bid price <= DA price
            # (So this should be nonincreasing, as bid price increases)
            is_in_the_money_matrix = bid_prices_row_vector <= dalmp_column_vector
        else:
            # True if bid price >= DA price
            # So this should be nondecreasing, as bid price increases
            is_in_the_money_matrix = bid_prices_row_vector >= dalmp_column_vector
        return is_in_the_money_matrix

    def precompute_cleared_spread_matrix(
        self, hour: int, is_sale: bool, bid_prices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute matrix: each row i corresponds to a scenario, each column j corresponds to a candidate DA bid price p_j.

        For is_sale=True, entry (i, j) is (price_DA_i - price_RT_i)*1{price_DA_i >= p_j)
        (i.e. the DART sale revenue from an offer at price p_j in scenario i)
        For is_sale=False, entry (i, j) is (price_RT_i - price_DA_i)*1{price_DA_i <= p_j)
        (i.e. the DART purchase cost from a bid at price p_j in scenario i)
        When multiplied by the column vector of decision variables, (the volume bid at each candidate price)
        and divided by n_scenarios, it gives the expected revenue for the hour under consideration.

        Args:
            hour: Hour to compute the spread matrix for
            is_sale: Flag indicating if this is for sell bids (True) or buy bids (False)
            bid_prices: Optional array of bid prices to use. If None, uses unique DA prices

        Returns:
            Matrix where each element (i,j) represents scenario revenue for scenario i at price j
        """
        # Ensure bid_prices is not None
        if bid_prices is None:
            bid_prices = self.market_data.get_unique_DA_prices_for_hour(hour)
        else:
            bid_prices = np.array(bid_prices)  # Ensure it's an ndarray for mypy joy

        ds = self.market_data.dataset
        dalmp = ds.sel(hour=hour)["dalmp"].values
        rtlmp = ds.sel(hour=hour)["rtlmp"].values

        # We could probably make this more efficient at the cost of readability
        # by allocating fewer intermediate matrices, but the optimization takes the bulk of the time anyway.
        bid_prices_row_vector = bid_prices[None, :]
        dalmp_column_vector = dalmp[:, None]
        rtlmp_column_vector = rtlmp[:, None]
        dart_spread_column_vector = (
            dalmp_column_vector - rtlmp_column_vector
        )  # shape (n_scenarios, 1)

        moneyness_bool_matrix = self.precompute_moneyness_bool_matrix(
            bid_prices_row_vector, dalmp_column_vector, is_sale
        )

        # Broadcast dart_spread to match moneyness_bool_matrix dimensions
        scenario_revenue_matrix = dart_spread_column_vector * moneyness_bool_matrix

        return scenario_revenue_matrix

    def print_bids_w_nonzero_volumes(self, w_value: np.ndarray, hour: int) -> None:
        """
        Print the DA prices and corresponding bid volumes for non-zero volumes
        
        Args:
            w_value: Array of bid volumes
            hour: Hour to get prices for
        """
        bid_prices = self.market_data.get_unique_DA_prices_for_hour(hour)
        
        for p, q in zip(bid_prices, w_value):
            if q > 0:
                print(f"DA price: {p}, bid volume: {q}MW,")

    def build_model(self):
        """
        Builds the CVXPY variables and constraints for each hour,
        plus CVaR constraints, volume caps, objective, etc.
        """
        ds = self.market_data.dataset
        constraints: List[cp.Constraint] = []

        # We'll accumulate scenario profits hour-by-hour
        scenario_profits = []

        # For volume caps across all hours
        total_sale_expr = 0
        total_buy_expr = 0

        # 1) For each hour, build the spread matrix and define the CP variables
        for hour_i in self.hours:
            # Retrieve candidate bid prices
            bid_prices = self.market_data.get_unique_DA_prices_for_hour(hour_i)
            self.bid_prices_per_hour.append(bid_prices)

            # Build the "scenario x price" matrix for sale or purchase
            sale_matrix = self.precompute_cleared_spread_matrix(hour_i, is_sale=True, bid_prices=bid_prices)
            buy_matrix = self.precompute_cleared_spread_matrix(hour_i, is_sale=False, bid_prices=bid_prices)

            n_scenarios, n_bids = sale_matrix.shape

            w_sell = cp.Variable(n_bids, nonneg=True, name=f"w_sell_hour{hour_i}")
            w_buy = cp.Variable(n_bids, nonneg=True, name=f"w_buy_hour{hour_i}")
            self.sell_vars.append(w_sell)
            self.buy_vars.append(w_buy)

            # Hourly scenario revenues
            sale_revenues = sale_matrix @ w_sell  # shape (n_scenarios,)
            purchase_cost = buy_matrix @ w_buy  # shape (n_scenarios,)
            # No wind revenues in the new model
            scenario_revenue_hour = sale_revenues - purchase_cost
            scenario_profits.append(scenario_revenue_hour)

            total_sale_expr += cp.sum(w_sell)
            total_buy_expr += cp.sum(w_buy)

        # 2) Volume constraints (sum over hours <= total cap)
        total_cap = self.max_bid_volume_per_hour * len(self.hours)
        constraints.append(total_sale_expr <= total_cap)  # type: ignore
        constraints.append(total_buy_expr <= total_cap)  # type: ignore

        # 3) Combine scenario profits across hours
        #    We want a shape-(n_scenarios,) expression for total daily profit
        # scenario_profits is a list of length(len(hours)), each an (n_scenarios,) expression
        # We can sum them directly: total_scenario_profit = sum(scenario_profits).
        total_scenario_profit = cp.sum(scenario_profits)  # shape (n_scenarios,)

        # 4) CVaR constraint
        n_scenarios = len(ds.scenario)
        t = cp.Variable(name="t_cvar")
        z = cp.Variable(n_scenarios, nonneg=True, name="z_cvar")

        constraints.append(z >= t - total_scenario_profit)
        constraints.append(t - (1.0 / ((1 - self.alpha) * n_scenarios)) * cp.sum(z) >= self.rho)

        # 5) Objective: maximize average profit
        sample_average_profit = cp.sum(total_scenario_profit) / n_scenarios
        objective = cp.Maximize(sample_average_profit)

        # 6) Create the CP problem
        self.problem = cp.Problem(objective, constraints)
        self.objective_expr = sample_average_profit

    def solve_model(self, solver=cp.CLARABEL):
        """
        Solve the built model and store results internally.
        
        Args:
            solver: CVXPY solver to use (default: cp.CLARABEL)
            
        Returns:
            Objective value (optimal expected revenue)
        """
        if self.problem is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        start_time = time.time()
        result = self.problem.solve(solver=solver, verbose=self.verbose)
        elapsed = time.time() - start_time
        self.objective_value = result

        if self.verbose:
            print(f"Solve completed in {elapsed:.2f} seconds. Status: {self.problem.status}")
            
        return result

    def get_solution(self) -> Tuple[
        List[NDArray[np.float64]],
        List[NDArray[np.float64]],
        float,
        List[NDArray[np.float64]],
    ]:
        """
        Return a list of sell decisions, buy decisions for each hour,
        the objective value, and the list of bid prices for each hour.
        
        Returns:
            Tuple containing:
            - List of sell decision vectors (one for each hour)
            - List of buy decision vectors (one for each hour)
            - Objective value (the optimal expected revenue)
            - List of bid price vectors (one for each hour)
        """
        if self.problem is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Warning: Problem not optimally solved. Status = {self.problem.status}")

        sell_decisions = []
        buy_decisions = []

        for w_s, w_b, bid_prices in zip(self.sell_vars, self.buy_vars, self.bid_prices_per_hour):
            if w_s.value is None:
                sell_decisions.append(np.zeros_like(bid_prices))
            else:
                sell_decisions.append(np.array(w_s.value))

            if w_b.value is None:
                buy_decisions.append(np.zeros_like(bid_prices))
            else:
                buy_decisions.append(np.array(w_b.value))

        # Make sure objective_value exists and is a float
        obj_value = 0.0 if self.objective_value is None else float(self.objective_value)

        return sell_decisions, buy_decisions, obj_value, self.bid_prices_per_hour

    def self_schedule_lower_bound(self):
        """
        Compute a lower bound on our method's objective value: expected revenue if we self-schedule
        the same quantity across all scenarios.
        
        Returns:
            Tuple: (objective value, buy decisions, sell decisions)
        """
        dataset = self.market_data.dataset
        hours = self.hours
        total_bid_volume_cap = self.max_bid_volume_per_hour * len(hours)
        
        n_hours = len(hours)
        n_scenarios = dataset.scenario.shape[0]
        purchase_decisions = cp.Variable(n_hours, nonneg=True)
        sale_decisions = cp.Variable(n_hours, nonneg=True)
        average_dalmp_by_hour = np.array(dataset["dalmp"].mean(dim="scenario").values)
        average_rtlmp_by_hour = np.array(dataset["rtlmp"].mean(dim="scenario").values)
        average_dart_spread_by_hour = average_dalmp_by_hour - average_rtlmp_by_hour
        constraints = [
            cp.sum(purchase_decisions) <= total_bid_volume_cap,
            cp.sum(sale_decisions) <= total_bid_volume_cap,
        ]
        
        # Objective function: maximize expected revenue (no wind revenue in the new model)
        expected_revenue = (
            cp.sum(
                average_dart_spread_by_hour @ (sale_decisions - purchase_decisions)
            )
            / n_scenarios
        )
        objective = cp.Maximize(expected_revenue)
        problem = cp.Problem(objective, constraints)
        problem.solve()
        assert (
            problem.status == "optimal"
        ), f"Expected optimal solution, got {problem.status}"
        return problem.value, purchase_decisions.value, sale_decisions.value

    def perfect_foresight_upper_bound(self):
        """
        Compute an upper bound: expected revenue if we had perfect foresight.
        I.e. we optimize each sample separately with self-scheduling.
        
        Returns:
            The upper bound as a float (expected revenue with perfect foresight)
        """
        dataset = self.market_data.dataset
        hours = self.hours
        max_bid_volume_per_hour = self.max_bid_volume_per_hour
        
        obj_vals = []
        for scenario in range(dataset.scenario.shape[0]):
            scenario_ds = dataset.sel(scenario=scenario)
            dalmps = scenario_ds["dalmp"].values
            rtlmps = scenario_ds["rtlmp"].values
            
            # Compute the DART spread for each hour
            dart_spread = dalmps - rtlmps
            sale_decisions = cp.Variable(len(hours), nonneg=True)
            purchase_decisions = cp.Variable(len(hours), nonneg=True)
            constraints = [
                cp.sum(sale_decisions) <= max_bid_volume_per_hour * len(hours),
                cp.sum(purchase_decisions) <= max_bid_volume_per_hour * len(hours),
            ]
            # No wind revenue in the new model
            revenue = (
                dart_spread @ (sale_decisions - purchase_decisions)
            )

            # Objective function: maximize revenue
            objective = cp.Maximize(cp.sum(revenue))
            problem = cp.Problem(objective, constraints)
            problem.solve()
            if problem.status != "optimal":
                raise Exception(f"Expected optimal solution, got {problem.status}")
                
            # Store the objective value for this scenario
            obj_vals.append(problem.value)
            
        return np.mean(obj_vals)
        
    def postprocess_bids(self) -> None:
        """
        After solve_model() is done, process the raw solution vectors into
        bid/offer information that can be analyzed or used for bidding.

        """
        # Get the solution vectors
        sell_decision_vectors, buy_decision_vectors, obj_val, bid_price_vectors = self.get_solution()

        self.finalized_bids = {}
        self.all_bids = {}
        
        # Iterate over each hour index
        for i, hour_i in enumerate(self.hours):
            w_sell = np.round(sell_decision_vectors[i], 1)
            w_buy = np.round(buy_decision_vectors[i], 1)
            prices = bid_price_vectors[i]

            # For backward compatibility, still keep track of the largest-volume bid
            # For the sell side, pick the largest-volume column
            if np.allclose(w_sell, 0.0):
                # No positive volumes => store None or zeros
                best_sell_price = np.nan
                best_sell_vol = 0.0
            else:
                sell_idx = np.argmax(w_sell)
                best_sell_price = prices[sell_idx]
                best_sell_vol = w_sell[sell_idx]

            # For the buy side
            if np.allclose(w_buy, 0.0):
                best_buy_price = np.nan
                best_buy_vol = 0.0
            else:
                buy_idx = np.argmax(w_buy)
                best_buy_price = prices[buy_idx]
                best_buy_vol = w_buy[buy_idx]

            # Store the largest-volume bids for backward compatibility
            self.finalized_bids[hour_i] = {
                "sell": {"price": float(best_sell_price), "volume_mw": float(best_sell_vol)},
                "buy": {"price": float(best_buy_price), "volume_mw": float(best_buy_vol)},
            }
            
            # Store ALL non-zero bids in the self.all_bids structure
            # Initialize the lists for this hour
            self.all_bids[hour_i] = {
                "sell": [],
                "buy": []
            }
            
            # Add all non-zero sell bids
            for j, (price, volume) in enumerate(zip(prices, w_sell)):
                if volume > 0.0:
                    self.all_bids[hour_i]["sell"].append({
                        "price": float(price),
                        "volume_mw": float(volume)
                    })
            
            # Add all non-zero buy bids
            for j, (price, volume) in enumerate(zip(prices, w_buy)):
                if volume > 0.0:
                    self.all_bids[hour_i]["buy"].append({
                        "price": float(price),
                        "volume_mw": float(volume)
                    })
            
            # Sort the bids by decreasing volume
            self.all_bids[hour_i]["sell"].sort(key=lambda x: x["volume_mw"], reverse=True)
            self.all_bids[hour_i]["buy"].sort(key=lambda x: x["volume_mw"], reverse=True)