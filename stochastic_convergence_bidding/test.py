import numpy as np

from stochastic_convergence_bidding.bidding_model import BiddingModel
from stochastic_convergence_bidding.market_data import MarketData
from stochastic_convergence_bidding.sample_data_generator import generate_sample_data

# test lower bound less than objective value less than upper bound

# test positive bid spread between buy and sell


def test_precompute_moneyness_matrices():
    """
    Confirm that the matrix of booleans representing whether a bid is in the money
    has correct structural properties
    """
    n_scenarios = 4
    hours = [0]
    data, target_names = generate_sample_data(
        num_samples=n_scenarios, num_hours=max(hours) + 1
    )
    market_data = MarketData(data, target_names)
    ds = market_data.get_xarray()
    
    # Create a bidding model
    model = BiddingModel(market_data, hours)

    # Copying setup code from precompute_cleared_spread_matrix() function...
    bid_prices = market_data.get_unique_DA_prices_for_hour(hours[0])
    dalmp = ds.sel(hour=hours[0])["dalmp"].values
    rtlmp = ds.sel(hour=hours[0])["rtlmp"].values

    bid_prices_row_vector = bid_prices[None, :]
    dalmp_column_vector = dalmp[:, None]

    sale_moneyness_matrix = model.precompute_moneyness_bool_matrix(
        bid_prices_row_vector, dalmp_column_vector, is_sale=True
    ).astype(float)

    for i in range(n_scenarios):
        # Nonincreasing: as the offer price gets higher, it eventually goes out of the money
        assert np.all(np.diff(sale_moneyness_matrix[i]) <= 0)

    purchase_moneyness_matrix = model.precompute_moneyness_bool_matrix(
        bid_prices_row_vector, dalmp_column_vector, is_sale=False
    )
    for i in range(n_scenarios):
        # Nondecreasing: as the bid price increases, it eventually goes in the money
        assert np.all(np.diff(purchase_moneyness_matrix[i]) >= 0)


def test_get_unique_DA_prices_for_hour():
    hours = [0]
    n_scenarios = 1000
    for i in range(10):
        data, target_names = generate_sample_data(
            num_samples=n_scenarios, num_hours=max(hours) + 1
        )
        market_data = MarketData(data, target_names)
        ds = market_data.get_xarray()

        n_unique_bid_prices = len(market_data.get_unique_DA_prices_for_hour(hours[0]))
        if n_unique_bid_prices < n_scenarios:
            # assert that there are duplicate DA prices in this hour
            assert len(np.unique(ds.sel(hour=0)["dalmp"].values)) < n_scenarios, (
                f"Expected fewer unique DA prices than scenarios, "
                f"found {len(np.unique(ds.sel(hour=0)['dalmp'].values))} "
                f"unique prices and {len(ds.scenario)} scenarios."
            )


def test_problem_lower_and_upper_bound():
    """
    Confirm that the problem objective value is greater than the lower bound based on the
    self-scheduling strategy that chooses the same purchase and sale quantities across all scenarios
    """
    data, target_names = generate_sample_data(num_samples=100, num_hours=24)
    market_data = MarketData(data, target_names)
    hours = list(range(24))
    
    # Create a bidding model with the given parameters
    model = BiddingModel(
        market_data=market_data,
        hours=hours,
        alpha=0.95,
        rho=-10000.0,
        max_bid_volume_per_hour=100,
        verbose=False
    )
    
    # Compute lower bound: self-scheduling the same purchases and sales across all scenarios
    lb_obj_val, lb_buy_decisions, lb_sale_decisions = model.self_schedule_lower_bound()
    
    # Compute upper bound: optimize purchase and sale decisions separately for each scenario
    ub_obj_val = model.perfect_foresight_upper_bound()
    
    # Run the stochastic optimization
    model.build_model(risk_constraint=True)
    model.solve_model()
    _, _, objective_value, _ = model.get_solution()
    
    assert (
        lb_obj_val <= objective_value
    ), f"Expected the lower bound objective value ({lb_obj_val:,.2f}) to be less than or equal to the actual objective value ({objective_value:,.2f})."

    assert (
        objective_value <= ub_obj_val
    ), f"Expected the objective value ({objective_value:,.2f}) to be less than or equal to the upper bound ({ub_obj_val:,.2f})."
