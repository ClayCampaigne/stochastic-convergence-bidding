import time
import numpy as np
import sys
import os
import concurrent.futures
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from loguru import logger

# Set max workers for parallel processing (adjust based on your CPU cores)
# Using N_CPU - 1 is a good rule of thumb to leave one core free for system tasks
MAX_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4

from stochastic_convergence_bidding.bidding_model import BiddingModel
from stochastic_convergence_bidding.market_data import MarketData
from stochastic_convergence_bidding.sample_data_generator import generate_sample_data

np.random.seed(1)

# Create output directory if it doesn't exist
os.makedirs("./results", exist_ok=True)

# Analysis parameters
sample_sizes = [500, 1000, 2000, 3000, 4000, 5000]  # Different scenario counts to test
num_bid_prices_list = [20, 50, None]  # Different number of bid prices to test (None = unlimited: use all
# sampled DA prices)

# Fixed grid of price points from -100 to 200 with $5 increments
fixed_grid_prices = list(range(-100, 205, 5))  # [-100, -95, -90, ..., 195, 200]
hours = list(range(24))

# Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"results/results_{timestamp}.txt"

# Cache for sample data to avoid regenerating in the same process
sample_data_cache = {}

# Function to run optimization with a specific sample size and number of bid prices
def run_optimization(n_scenarios, num_bid_prices=None, verbose=False, is_risk_constraint=True, fixed_bid_prices=None):
    """
    Run optimization with the specified parameters.
    
    Args:
        n_scenarios: Number of scenarios to use
        num_bid_prices: Number of price points to use (None = all unique prices from samples)
        verbose: Whether to print progress information
        is_risk_constraint: Whether to apply the risk constraint (CVaR)
        fixed_bid_prices: List of fixed price points to use. Cannot be combined with num_bid_prices.
        
    Returns:
        Tuple of (objective_value, elapsed_time, model)
    """
    # Ensure we're not using both num_bid_prices and fixed_bid_prices
    if num_bid_prices is not None and fixed_bid_prices is not None:
        raise ValueError("Cannot specify both num_bid_prices and fixed_bid_prices. Choose one approach.")
    
    # Use cached data if available for this process and scenario count
    pid = os.getpid()
    cache_key = (pid, n_scenarios)
    
    if cache_key in sample_data_cache:
        data, target_names = sample_data_cache[cache_key]
    else:
        # Generate sample data
        data, target_names = generate_sample_data(
            num_samples=n_scenarios, num_hours=max(hours) + 1, random_seed=0
        )
        # Cache the data for this process and scenario count
        sample_data_cache[cache_key] = (data, target_names)
    
    # Create market data with the appropriate price configuration
    market_data = MarketData(data, target_names, fixed_grid_prices=fixed_bid_prices)
    
    # Print hourly statistics report if verbose
    if verbose:
        market_data.print_hourly_report()
    
    # Create and solve the Bidding Model
    model = BiddingModel(
        market_data=market_data,
        hours=hours,
        alpha=0.95,
        rho=-1000.0,
        verbose=False
    )
    
    # Pass the num_bid_prices parameter to the MarketData's get_DA_bid_prices method
    # This will be used internally by the BiddingModel when it calls market_data.get_DA_bid_prices
    market_data.num_bid_prices = num_bid_prices

    start_time = time.time()
    model.build_model(is_risk_constraint)
    model.solve_model()
    model.postprocess_bids()
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Expected Revenue: ${model.objective_value:.2f}")
    
    return model.objective_value, elapsed_time, model


# Function to run the main optimization and print detailed results
def run_optimization_and_report_results(n_scenarios, is_risk_constraint=True, use_fixed_grid=False,
                                        num_bid_prices=None):
    # Configure loguru to output to both console and file
    # First, remove any existing handlers
    logger.remove()
    
    # Add console handler
    logger.add(sys.stdout, level="INFO")
    
    # Add file handler with 'w' mode to overwrite any existing file
    logger.add(output_file, level="INFO", mode="w")
    
    # Print a header
    logger.info(f"Optimization Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"=" * 80)
    
    logger.info(f"\n\n{'=' * 40}")
    logger.info(f"Running optimization for {n_scenarios} scenarios...")
    logger.info(f"{'=' * 40}\n")

    start_time = time.time()
    
    # Set up parameters for fixed grid or normal optimization
    fixed_bid_prices = fixed_grid_prices if use_fixed_grid else None
    
    # Run the optimization
    revenue, solve_time, model = run_optimization(
        n_scenarios, 
        verbose=True, 
        is_risk_constraint=is_risk_constraint,
        fixed_bid_prices=fixed_bid_prices
    )
    
    end_time = time.time()
    logger.info(f"\nTotal time (including data generation): {end_time - start_time:.2f} seconds")
    logger.info(f"Expected Revenue: ${revenue:.2f}")
    
    # Print all bids for each hour
    logger.info("\nBids for each hour:")
    active_hours = []
    
    for hour_j in hours:
        all_bids = model.all_bids[hour_j]
        
        # Skip hours with no activity
        sell_bids = all_bids["sell"]
        buy_bids = all_bids["buy"]
        
        if not sell_bids and not buy_bids:
            continue
            
        active_hours.append(hour_j)
        logger.info(f"\nHour {hour_j}:")
        
        # Sell side
        if sell_bids:
            logger.info(f"  Sell bids:")
            for i, bid in enumerate(sell_bids):
                logger.info(f"    #{i+1}: DA price=${bid['price']:.2f}/MWh, volume={bid['volume_mw']:.1f} MW")
        else:
            logger.info("  No sell bids")
        
        # Buy side
        if buy_bids:
            logger.info(f"  Buy bids:")
            for i, bid in enumerate(buy_bids):
                logger.info(f"    #{i+1}: DA price=${bid['price']:.2f}/MWh, volume={bid['volume_mw']:.1f} MW")
        else:
            logger.info("  No buy bids")
    
    # Summary of active hours
    if active_hours:
        logger.info("\nSummary of active hours:")
        for hour in sorted(active_hours):
            all_bids = model.all_bids[hour]
            sell_bids = all_bids["sell"]
            buy_bids = all_bids["buy"]
            
            # Calculate the total volume for sell and buy
            total_sell_volume = sum(bid["volume_mw"] for bid in sell_bids)
            total_buy_volume = sum(bid["volume_mw"] for bid in buy_bids)
            
            # Count the number of non-zero bids
            num_sell_bids = len(sell_bids)
            num_buy_bids = len(buy_bids)
            
            logger.info(f"  Hour {hour}: {num_sell_bids} sell bid(s) with total volume = {total_sell_volume:.1f}MW, " 
                  f"{num_buy_bids} buy bid(s) with total volume = {total_buy_volume:.1f}MW")
    else:
        logger.info("\nNo active bids in any hour.")
    
    # Write a clear separator to mark completion
    logger.info(f"\n{'=' * 40}")
    logger.info(f"COMPLETED: {n_scenarios} scenarios")
    logger.info(f"{'=' * 40}")
    
    # Remove all handlers to clean up resources
    logger.remove()
    
    return revenue, solve_time

# Function to run a single analysis case (for parallel execution)
def run_single_analysis_case(n_scenarios, num_bid_prices, results_dir, risk_constraint=True, use_fixed_grid=False):
    """Run a single analysis case for a specific number of scenarios and price points.
    
    Args:
        n_scenarios: Number of scenarios to use
        num_bid_prices: Number of price points to use (None = unlimited)
        results_dir: Directory to save interim results
        risk_constraint: Whether to apply the risk constraint (CVaR) in the model
        use_fixed_grid: Whether to use the fixed grid of price points
        
    Returns:
        Tuple of (n_scenarios, num_bid_prices, revenue, solve_time, total_time)
    """
    # Create a process and case specific ID
    if use_fixed_grid:
        price_points_str = "fixed_grid"
        fixed_bid_prices = fixed_grid_prices  # Use the global fixed grid
        num_bid_prices_arg = None
    else:
        price_points_str = str(num_bid_prices) if num_bid_prices is not None else "unlimited"
        fixed_bid_prices = None
        num_bid_prices_arg = num_bid_prices
    
    risk_str = "with risk" if risk_constraint else "no risk"
    pid = os.getpid()
    
    grid_str = "fixed grid" if use_fixed_grid else price_points_str
    print(f"[Process {pid}] Running with {n_scenarios} scenarios and {grid_str} price points ({risk_str})...")
    
    start_time = time.time()
    # Run the optimization
    revenue, solve_time, _ = run_optimization(
        n_scenarios, 
        num_bid_prices=num_bid_prices_arg, 
        verbose=False, 
        is_risk_constraint=risk_constraint,
        fixed_bid_prices=fixed_bid_prices
    )
    total_time = time.time() - start_time
    
    # Create a results string
    risk_info = "with risk constraint" if risk_constraint else "without risk constraint"
    results_str = (f"{n_scenarios} scenarios, {price_points_str} price points ({risk_info}):\n"
                  f"  Revenue: ${revenue:.2f}, Solve time: {solve_time:.2f}s, Total time: {total_time:.2f}s\n\n")
    
    print(f"[Process {pid}] Completed: {n_scenarios} scenarios, {price_points_str} price points ({risk_str}) - "
          f"Revenue: ${revenue:.2f}, Solve time: {solve_time:.2f}s, Total time: {total_time:.2f}s")
    
    # Save individual result to a process-specific file
    risk_label = "risk" if risk_constraint else "norisk"
    case_file = f"{results_dir}/case_{n_scenarios}_{price_points_str}_{risk_label}_{pid}.txt"
    with open(case_file, 'w') as f:
        f.write(results_str)
    
    return (n_scenarios, num_bid_prices, revenue, solve_time, total_time)

# Run the analysis of bid prices vs. revenue for different sample sizes
def run_bid_price_analysis(risk_constraint=True, use_fixed_grid=False):
    results = {}
    for n_scenarios in sample_sizes:
        results[n_scenarios] = []
    
    risk_str = "with risk constraint" if risk_constraint else "without risk constraint"
    grid_str = "using fixed price grid" if use_fixed_grid else "using sample-based price points"
    print(f"\nRunning analysis of price points vs. revenue with parallelization ({risk_str}, {grid_str})...")
    
    # Create a directory for interim results
    risk_label = "risk" if risk_constraint else "norisk"
    grid_label = "fixedgrid" if use_fixed_grid else "samplegrid"
    interim_dir = f"results/interim_{timestamp}_{risk_label}_{grid_label}"
    os.makedirs(interim_dir, exist_ok=True)
    
    # Create a summary file
    summary_file = f"{interim_dir}/summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Price Points Analysis - Summary\n")
        f.write("============================\n\n")
        f.write(f"Using {MAX_WORKERS} parallel workers\n")
        f.write(f"Risk constraint: {'Enabled' if risk_constraint else 'Disabled'}\n")
        f.write(f"Price grid: {'Fixed grid (-100 to 200, $5 increments)' if use_fixed_grid else 'Sample-based'}\n\n")
        f.write("Analysis cases to run:\n")
        if use_fixed_grid:
            # If using fixed grid, only show different scenario counts
            for n_scenarios in sample_sizes:
                f.write(f"- {n_scenarios} scenarios, fixed grid price points\n")
        else:
            # Otherwise, show combinations of scenarios and price points
            for n_scenarios in sample_sizes:
                for num_bid_prices in num_bid_prices_list:
                    # Skip cases where num_bid_prices exceeds sample size
                    if num_bid_prices is not None and num_bid_prices >= n_scenarios:
                        continue
                    price_points_str = str(num_bid_prices) if num_bid_prices is not None else "unlimited"
                    f.write(f"- {n_scenarios} scenarios, {price_points_str} price points\n")
        f.write("\n")
    
    # Create a list of all analysis cases to run
    analysis_cases = []
    
    if use_fixed_grid:
        # If using fixed grid, just analyze different scenario counts with fixed grid
        for n_scenarios in sample_sizes:
            analysis_cases.append((n_scenarios, None))  # num_bid_prices is not used with fixed grid
    else:
        # Otherwise, analyze different combinations of scenarios and price points
        for n_scenarios in sample_sizes:
            for num_bid_prices in num_bid_prices_list:
                # Skip cases where num_bid_prices exceeds sample size
                if num_bid_prices is not None and num_bid_prices >= n_scenarios:
                    continue
                analysis_cases.append((n_scenarios, num_bid_prices))
    
    # Run the analysis cases in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks and store futures
        futures = [executor.submit(run_single_analysis_case, n, p, interim_dir, risk_constraint, use_fixed_grid) 
                   for n, p in analysis_cases]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                n_scenarios, num_bid_prices, revenue, solve_time, total_time = future.result()
                results[n_scenarios].append((num_bid_prices, revenue, solve_time))
            except Exception as e:
                print(f"Error in analysis case: {e}")
    
    # Sort the results for each scenario count by price points order
    for n_scenarios in sample_sizes:
        # Sort the available results by price points
        results[n_scenarios].sort(key=lambda x: (0 if x[0] is None else x[0]))
    
    # Combine all individual result files into a single interim results file
    with open(f"results/interim_results_{timestamp}_{risk_label}.txt", 'w') as f:
        f.write("Price Points Analysis - Interim Results\n")
        f.write("=====================================\n\n")
        
        # Append all individual case files in order
        for n_scenarios in sample_sizes:
            for num_bid_prices in num_bid_prices_list:
                price_points_str = str(num_bid_prices) if num_bid_prices is not None else "unlimited"
                pattern = f"{interim_dir}/case_{n_scenarios}_{price_points_str}_{risk_label}_*.txt"
                case_files = os.listdir(interim_dir)
                
                # Find matching case file
                matching_files = [file for file in case_files 
                                 if file.startswith(f"case_{n_scenarios}_{price_points_str}_{risk_label}_")]
                
                if matching_files:
                    case_file = os.path.join(interim_dir, matching_files[0])
                    with open(case_file, 'r') as case_f:
                        f.write(case_f.read())
    
    # Create figures for the analysis
    fig1 = plt.figure(figsize=(10, 6))
    
    if use_fixed_grid:
        # For fixed grid, we just plot revenue vs sample size
        # Sort sample sizes for cleaner plot
        sorted_sample_sizes = sorted(sample_sizes)
        
        # Get revenues for each sample size
        revenues = []
        for n_scenarios in sorted_sample_sizes:
            if results[n_scenarios]:  # Check if we have results for this scenario count
                revenues.append(results[n_scenarios][0][1])  # [0][1] is the revenue for the first price point
            else:
                revenues.append(0)  # Use 0 as placeholder if no results
        
        # Create the plot
        plt.figure(fig1.number)
        plt.plot(sorted_sample_sizes, revenues, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Add data labels
        for i, (x, y) in enumerate(zip(sorted_sample_sizes, revenues)):
            if y > 0:  # Only label non-zero values
                plt.annotate(f"${y:.2f}", (x, y), textcoords="offset points", 
                             xytext=(0, 10), ha='center', fontsize=10)
    else:
        # Original code for price points analysis
        # Create x values for plotting (replace None with "unlimited" for display)
        x_labels = [str(x) if x is not None else "unlimited" for x in num_bid_prices_list]
        x_numeric = list(range(len(num_bid_prices_list)))
        
        # Create first plot: Revenue vs Price Points
        plt.figure(fig1.number)
        for n_scenarios in sample_sizes:
            # Create x-y pairs for available data points
            points = []
            for i, price_point in enumerate(num_bid_prices_list):
                # Find this price point in the results
                result_entry = next((res for res in results[n_scenarios] if res[0] == price_point), None)
                if result_entry:
                    points.append((i, result_entry[1]))  # (x_position, revenue)
            
            # Sort points by x position
            points.sort(key=lambda p: p[0])
            
            # Extract x and y values for plotting
            if points:
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                plt.plot(x_values, y_values, marker='o', label=f"{n_scenarios} scenarios")
    
    # Add labels and legend
    if use_fixed_grid:
        plt.xlabel("Number of Scenarios", fontsize=12)
        plt.ylabel("Expected Revenue ($)", fontsize=12)
        plt.title("Expected Revenue vs. Number of Scenarios (Fixed Price Grid)", fontsize=14)
    else:
        plt.xticks(x_numeric, x_labels)
        plt.xlabel("Number of Price Points", fontsize=12)
        plt.ylabel("Expected Revenue ($)", fontsize=12)
        plt.title("Expected Revenue vs. Number of Price Points", fontsize=14)
        plt.legend()
    plt.grid(True)
    
    # Save the first figure
    risk_label = "risk" if risk_constraint else "norisk"
    grid_label = "fixedgrid" if use_fixed_grid else "samplegrid"
    revenue_analysis_file = f"results/revenue_vs_price_points_{timestamp}_{risk_label}_{grid_label}.png"
    plt.savefig(revenue_analysis_file)
    
    # Create second plot: Solution Time vs Price Points
    fig2 = plt.figure(figsize=(10, 6))
    
    if use_fixed_grid:
        # For fixed grid, we just plot solution time vs sample size
        # Sort sample sizes for cleaner plot
        sorted_sample_sizes = sorted(sample_sizes)
        
        # Get solve times for each sample size
        solve_times = []
        for n_scenarios in sorted_sample_sizes:
            if results[n_scenarios]:  # Check if we have results for this scenario count
                solve_times.append(results[n_scenarios][0][2])  # [0][2] is the solve time for the first price point
            else:
                solve_times.append(0)  # Use 0 as placeholder if no results
        
        # Create the plot
        plt.figure(fig2.number)
        plt.plot(sorted_sample_sizes, solve_times, marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Add data labels
        for i, (x, y) in enumerate(zip(sorted_sample_sizes, solve_times)):
            if y > 0:  # Only label non-zero values
                plt.annotate(f"{y:.2f}s", (x, y), textcoords="offset points", 
                             xytext=(0, 10), ha='center', fontsize=10)
        
        # Add labels and legend
        plt.xlabel("Number of Scenarios", fontsize=12)
        plt.ylabel("Solution Time (seconds)", fontsize=12)
        plt.title("Solution Time vs. Number of Scenarios (Fixed Price Grid)", fontsize=14)
    else:
        # Original code for price points analysis
        for n_scenarios in sample_sizes:
            # Create x-y pairs for available data points
            points = []
            for i, price_point in enumerate(num_bid_prices_list):
                # Find this price point in the results
                result_entry = next((res for res in results[n_scenarios] if res[0] == price_point), None)
                if result_entry:
                    points.append((i, result_entry[2]))  # (x_position, solve_time)
            
            # Sort points by x position
            points.sort(key=lambda p: p[0])
            
            # Extract x and y values for plotting
            if points:
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                plt.plot(x_values, y_values, marker='o', label=f"{n_scenarios} scenarios")
        
        # Add labels and legend
        plt.xticks(x_numeric, x_labels)
        plt.xlabel("Number of Price Points", fontsize=12)
        plt.ylabel("Solution Time (seconds)", fontsize=12)
        plt.title("Solution Time vs. Number of Price Points", fontsize=14)
        plt.legend()
    
    plt.grid(True)
    
    # Save the second figure
    time_analysis_file = f"results/time_vs_price_points_{timestamp}_{risk_label}_{grid_label}.png"
    plt.savefig(time_analysis_file)
    
    # Create third plot: Revenue vs Solution Time (tradeoff plot)
    fig3 = plt.figure(figsize=(10, 6))
    
    if use_fixed_grid:
        # For fixed grid, we make a scatter plot with annotations of each scenario count
        # Get data points for the plot
        points = []
        for n_scenarios in sample_sizes:
            if results[n_scenarios]:  # Check if we have results for this scenario count
                solve_time = results[n_scenarios][0][2]  # Solve time
                revenue = results[n_scenarios][0][1]  # Revenue
                points.append((n_scenarios, solve_time, revenue))
        
        if points:
            # Sort points by solve time for clarity
            points.sort(key=lambda p: p[1])
            
            # Extract x and y values
            x_values = [p[1] for p in points]  # Solve times on x-axis
            y_values = [p[2] for p in points]  # Revenues on y-axis
            
            # Create scatter plot
            plt.scatter(x_values, y_values, s=100)
            
            # Add trend line
            plt.plot(x_values, y_values, linestyle='--', alpha=0.5)
            
            # Annotate each point with scenario count
            for i, (n_scenarios, solve_time, revenue) in enumerate(points):
                plt.annotate(f"{n_scenarios} scenarios", 
                             (solve_time, revenue),
                             textcoords="offset points", 
                             xytext=(5, 5), 
                             fontsize=10)
        
        # Add labels and title
        plt.xlabel("Solution Time (seconds)", fontsize=12)
        plt.ylabel("Expected Revenue ($)", fontsize=12)
        plt.title("Revenue vs. Solution Time Tradeoff (Fixed Price Grid)", fontsize=14)
    else:
        # Original code for price points analysis
        for n_scenarios in sample_sizes:
            # Get available results for this scenario count
            scenario_results = results[n_scenarios]
            
            if scenario_results:
                # Extract x and y values from available results
                x_values = [res[2] for res in scenario_results]  # Solution times
                y_values = [res[1] for res in scenario_results]  # Revenues
                
                # Add the line and markers
                plt.plot(x_values, y_values, marker='o', label=f"{n_scenarios} scenarios")
                
                # Annotate each point with its price points
                for price_points, revenue, solve_time in scenario_results:
                    price_label = str(price_points) if price_points is not None else "Unl."
                    plt.annotate(price_label, (solve_time, revenue), 
                                textcoords="offset points", xytext=(0,10), ha='center')
        
        # Add labels and legend
        plt.xlabel("Solution Time (seconds)", fontsize=12)
        plt.ylabel("Expected Revenue ($)", fontsize=12)
        plt.title("Revenue vs. Solution Time Tradeoff", fontsize=14)
        plt.legend()
    
    plt.grid(True)
    
    # Save the third figure
    tradeoff_analysis_file = f"results/revenue_time_tradeoff_{timestamp}_{risk_label}_{grid_label}.png"
    plt.savefig(tradeoff_analysis_file)
    
    # Generate a summary table
    with open(f"results/price_points_analysis_{timestamp}_{risk_label}_{grid_label}.txt", 'w') as f:
        if use_fixed_grid:
            f.write("Fixed Grid Price Points Analysis Results\n")
        else:
            f.write("Price Points Analysis Results\n")
        f.write("==========================\n\n")
        
        # Table header
        if use_fixed_grid:
            f.write(f"{'Sample Size':<15}")
        else:
            f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        # Table rows for revenue
        if use_fixed_grid:
            # For fixed grid, we just have one row with revenue for each sample size
            f.write(f"{'Fixed Grid':<15}")
            for n_scenarios in sample_sizes:
                if results[n_scenarios]:
                    revenue = results[n_scenarios][0][1]  # First result for this scenario count
                    f.write(f"${revenue:>23.2f}")
                else:
                    f.write(f"{'N/A':>24}")
            f.write("\n")
        else:
            # For regular price points analysis
            for i, num_bid_prices in enumerate(num_bid_prices_list):
                price_label = str(num_bid_prices) if num_bid_prices is not None else "unlimited"
                f.write(f"{price_label:<15}")
                for n_scenarios in sample_sizes:
                    # Find this price point in the results for this scenario count
                    result_entry = next((res for res in results[n_scenarios] if res[0] == num_bid_prices), None)
                    
                    if result_entry:
                        revenue = result_entry[1]
                        f.write(f"${revenue:>23.2f}")
                    else:
                        # This case was skipped
                        f.write(f"{'N/A':>24}")
                f.write("\n")
        
        # Add revenue improvement percentage vs 5 price points baseline
        f.write("\nRevenue Improvement % (vs. 5 price points):\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        # Get the baseline price point (first in the list)
        baseline_price_point = num_bid_prices_list[0]
        
        # Calculate improvement percentages
        for i, num_bid_prices in enumerate(num_bid_prices_list):
            price_label = str(num_bid_prices) if num_bid_prices is not None else "unlimited"
            f.write(f"{price_label:<15}")
            
            for n_scenarios in sample_sizes:
                # Find baseline result for this scenario count
                baseline_entry = next((res for res in results[n_scenarios] if res[0] == baseline_price_point), None)
                
                # Find this price point in the results for this scenario count
                current_entry = next((res for res in results[n_scenarios] if res[0] == num_bid_prices), None)
                
                if num_bid_prices == baseline_price_point:
                    # This is the baseline
                    f.write(f"{'baseline':>23}")
                elif baseline_entry and current_entry:
                    # Calculate improvement percentage
                    baseline_revenue = baseline_entry[1]
                    current_revenue = current_entry[1]
                    improvement_pct = (current_revenue - baseline_revenue) / baseline_revenue * 100
                    f.write(f"{improvement_pct:>23.2f}%")
                else:
                    # This case was skipped
                    f.write(f"{'N/A':>24}")
            f.write("\n")
        
        # Add solve times
        f.write("\nSolve Times (seconds):\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        for i, num_bid_prices in enumerate(num_bid_prices_list):
            price_label = str(num_bid_prices) if num_bid_prices is not None else "unlimited"
            f.write(f"{price_label:<15}")
            for n_scenarios in sample_sizes:
                # Find this price point in the results for this scenario count
                result_entry = next((res for res in results[n_scenarios] if res[0] == num_bid_prices), None)
                
                if result_entry:
                    solve_time = result_entry[2]
                    f.write(f"{solve_time:>23.2f}")
                else:
                    # This case was skipped
                    f.write(f"{'N/A':>24}")
            f.write("\n")
        
        # Add solve time increase ratio vs 5 price points
        f.write("\nSolve Time Ratio (vs. 5 price points):\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        # Get the baseline price point (first in the list)
        baseline_price_point = num_bid_prices_list[0]
        
        for i, num_bid_prices in enumerate(num_bid_prices_list):
            price_label = str(num_bid_prices) if num_bid_prices is not None else "unlimited"
            f.write(f"{price_label:<15}")
            
            for n_scenarios in sample_sizes:
                # Find baseline result for this scenario count
                baseline_entry = next((res for res in results[n_scenarios] if res[0] == baseline_price_point), None)
                
                # Find this price point in the results for this scenario count
                current_entry = next((res for res in results[n_scenarios] if res[0] == num_bid_prices), None)
                
                if num_bid_prices == baseline_price_point:
                    # This is the baseline
                    f.write(f"{'baseline':>23}")
                elif baseline_entry and current_entry:
                    # Calculate time ratio
                    baseline_time = baseline_entry[2]
                    current_time = current_entry[2]
                    time_ratio = current_time / baseline_time
                    f.write(f"{time_ratio:>23.2f}x")
                else:
                    # This case was skipped
                    f.write(f"{'N/A':>24}")
            f.write("\n")
    
    print(f"\nAnalysis complete.")
    print(f"- Revenue vs Price Points plot: {revenue_analysis_file}")
    print(f"- Solution Time vs Price Points plot: {time_analysis_file}")
    print(f"- Revenue vs Solution Time tradeoff plot: {tradeoff_analysis_file}")
    print(f"- Detailed results table: results/price_points_analysis_{timestamp}_{risk_label}_{grid_label}.txt")
    
    return results

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run stochastic convergence bidding analysis")
    parser.add_argument("--analysis", action="store_true", help="Run price points analysis")
    parser.add_argument("--scenarios", type=int, default=100, help="Number of scenarios for main run")
    parser.add_argument("--num-price-points", type=int, help="Number of price points to use (evenly spaced). If not specified, all unique prices are used.")
    parser.add_argument("--no-risk-constraint", action="store_true", 
                        help="Disable the risk constraint (CVaR). Default is to use the risk constraint.")
    parser.add_argument("--fixed-grid", action="store_true",
                        help="Use a fixed grid of price points (-100 to 200, $5 increments) instead of sample-based prices.")
    args = parser.parse_args()
    
    # Use risk constraint unless --no-risk-constraint flag is provided
    risk_constraint = not args.no_risk_constraint
    use_fixed_grid = args.fixed_grid
    num_bid_prices = args.num_price_points  # This will be None if not specified
    
    if args.analysis:
        # Run the analysis of price points vs. revenue
        results = run_bid_price_analysis(
            risk_constraint=risk_constraint,
            use_fixed_grid=use_fixed_grid
        )
    else:
        # Run the standard optimization to get detailed output
        revenue, solve_time = run_optimization_and_report_results(
            args.scenarios, 
            is_risk_constraint=risk_constraint,
            use_fixed_grid=use_fixed_grid,
            num_bid_prices=num_bid_prices
        )
        risk_str = "with risk constraint" if risk_constraint else "without risk constraint"
        
        if use_fixed_grid:
            grid_str = "using fixed price grid"
        elif num_bid_prices is not None:
            grid_str = f"using {num_bid_prices} evenly-spaced price points"
        else:
            grid_str = "using all unique sample prices"
            
        print(f"Results saved to {output_file} ({risk_str}, {grid_str})")