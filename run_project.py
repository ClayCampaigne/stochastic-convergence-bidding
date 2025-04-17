import time
import numpy as np
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

from stochastic_convergence_bidding.bidding_model import BiddingModel
from stochastic_convergence_bidding.market_data import MarketData
from stochastic_convergence_bidding.sample_data_generator import generate_sample_data

np.random.seed(1)

# Create output directory if it doesn't exist
os.makedirs("./results", exist_ok=True)

# Analysis parameters
# Full analysis parameters
sample_sizes = [100, 500, 1000]  # Different scenario counts to test
max_bid_prices_list = [5, 10, 20, 50, 100, None]  # Different max bid prices to test (None = unlimited)
hours = list(range(24))

# Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"results/results_{timestamp}.txt"

# Create a custom MarketData class that can limit the number of bid prices
class LimitedPricePointsMarketData(MarketData):
    def __init__(self, data, target_names, max_bid_prices=None):
        super().__init__(data, target_names)
        self.max_bid_prices = max_bid_prices
        # Store the original method
        self.original_get_unique_DA_prices = super().get_unique_DA_prices_for_hour
    
    def get_unique_DA_prices_for_hour(self, hour):
        if self.max_bid_prices is None:
            # Use the original implementation
            return self.original_get_unique_DA_prices(hour)
        else:
            # Get the min and max prices, then create evenly spaced price points
            hour_data = self.dataset.sel(hour=hour)
            min_price = np.min(hour_data["dalmp"].values)
            max_price = np.max(hour_data["dalmp"].values)
            
            # Generate evenly spaced price points
            return np.linspace(min_price, max_price, self.max_bid_prices)

# Function to run optimization with a specific sample size and max bid prices
def run_optimization(n_scenarios, max_bid_prices=None, verbose=False):
    # Generate sample data
    data, target_names = generate_sample_data(
        num_samples=n_scenarios, num_hours=max(hours) + 1, random_seed=0
    )
    
    # Create market data with optional price point limitation
    market_data = LimitedPricePointsMarketData(data, target_names, max_bid_prices)
    
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

    start_time = time.time()
    model.build_model()
    model.solve_model()
    model.postprocess_bids()
    elapsed_time = time.time() - start_time
    
    if verbose:
        print(f"Optimization completed in {elapsed_time:.2f} seconds")
        print(f"Expected Revenue: ${model.objective_value:.2f}")
    
    return model.objective_value, elapsed_time, model

# Function to run the main optimization and print detailed results
def run_main_optimization(n_scenarios):
    # Redirect output to both console and file
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        class Tee:
            def __init__(self, *files):
                self.files = files
            
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()  # Ensure immediate write
            
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = Tee(sys.stdout, f)
        
        # Print a header
        print(f"Optimization Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 80)
        
        print(f"\n\n{'=' * 40}")
        print(f"Running optimization for {n_scenarios} scenarios...")
        print(f"{'=' * 40}\n")
    
        start_time = time.time()
        
        # Run the optimization
        revenue, solve_time, model = run_optimization(n_scenarios, verbose=True)
        
        end_time = time.time()
        print(f"\nTotal time (including data generation): {end_time - start_time:.2f} seconds")
        print(f"Expected Revenue: ${revenue:.2f}")
        
        # Flush output to ensure results are saved
        sys.stdout.flush()
    
        # Print all bids for each hour
        print("\nBids for each hour:")
        active_hours = []
        
        for hour_j in hours:
            all_bids = model.all_bids[hour_j]
            
            # Skip hours with no activity
            sell_bids = all_bids["sell"]
            buy_bids = all_bids["buy"]
            
            if not sell_bids and not buy_bids:
                continue
                
            active_hours.append(hour_j)
            print(f"\nHour {hour_j}:")
            
            # Sell side
            if sell_bids:
                print(f"  Sell bids:")
                for i, bid in enumerate(sell_bids):
                    print(f"    #{i+1}: DA price=${bid['price']:.2f}/MWh, volume={bid['volume_mw']:.1f} MW")
            else:
                print("  No sell bids")
            
            # Buy side
            if buy_bids:
                print(f"  Buy bids:")
                for i, bid in enumerate(buy_bids):
                    print(f"    #{i+1}: DA price=${bid['price']:.2f}/MWh, volume={bid['volume_mw']:.1f} MW")
            else:
                print("  No buy bids")
            
            # Flush after each hour's output
            sys.stdout.flush()
        
        # Summary of active hours
        if active_hours:
            print("\nSummary of active hours:")
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
                
                print(f"  Hour {hour}: {num_sell_bids} sell bid(s) with total volume = {total_sell_volume:.1f}MW, " 
                      f"{num_buy_bids} buy bid(s) with total volume = {total_buy_volume:.1f}MW")
        else:
            print("\nNo active bids in any hour.")
            
        # Flush at the end of iteration
        sys.stdout.flush()
        
        # Write a clear separator to mark completion
        print(f"\n{'=' * 40}")
        print(f"COMPLETED: {n_scenarios} scenarios")
        print(f"{'=' * 40}")
        sys.stdout.flush()
    
        # Restore original stdout
        sys.stdout = original_stdout
    
    return revenue, solve_time

# Run the analysis of bid prices vs. revenue for different sample sizes
def run_bid_price_analysis():
    results = {}
    
    print("\nRunning analysis of price points vs. revenue...")
    
    # Create figures for the analysis
    fig1 = plt.figure(figsize=(10, 6))
    
    # For each sample size
    for n_scenarios in sample_sizes:
        results[n_scenarios] = []
        
        # For each max bid prices setting
        for max_bid_prices in max_bid_prices_list:
            print(f"Running with {n_scenarios} scenarios and {max_bid_prices if max_bid_prices is not None else 'unlimited'} price points...")
            
            # Run the optimization
            revenue, solve_time, _ = run_optimization(n_scenarios, max_bid_prices)
            
            # Store the results
            results[n_scenarios].append((max_bid_prices, revenue, solve_time))
            
            price_points_str = str(max_bid_prices) if max_bid_prices is not None else "unlimited"
            print(f"  Revenue: ${revenue:.2f}, Solve time: {solve_time:.2f}s")
        
    # Create x values for plotting (replace None with "unlimited" for display)
    x_labels = [str(x) if x is not None else "unlimited" for x in max_bid_prices_list]
    x_numeric = list(range(len(max_bid_prices_list)))
    
    # Create first plot: Revenue vs Price Points
    plt.figure(fig1.number)
    for n_scenarios in sample_sizes:
        y_values = [x[1] for x in results[n_scenarios]]
        plt.plot(x_numeric, y_values, marker='o', label=f"{n_scenarios} scenarios")
    
    # Add labels and legend
    plt.xticks(x_numeric, x_labels)
    plt.xlabel("Number of Price Points")
    plt.ylabel("Expected Revenue ($)")
    plt.title("Expected Revenue vs. Number of Price Points")
    plt.legend()
    plt.grid(True)
    
    # Save the first figure
    revenue_analysis_file = f"results/revenue_vs_price_points_{timestamp}.png"
    plt.savefig(revenue_analysis_file)
    
    # Create second plot: Solution Time vs Price Points
    fig2 = plt.figure(figsize=(10, 6))
    for n_scenarios in sample_sizes:
        y_values = [x[2] for x in results[n_scenarios]]
        plt.plot(x_numeric, y_values, marker='o', label=f"{n_scenarios} scenarios")
    
    # Add labels and legend
    plt.xticks(x_numeric, x_labels)
    plt.xlabel("Number of Price Points")
    plt.ylabel("Solution Time (seconds)")
    plt.title("Solution Time vs. Number of Price Points")
    plt.legend()
    plt.grid(True)
    
    # Save the second figure
    time_analysis_file = f"results/time_vs_price_points_{timestamp}.png"
    plt.savefig(time_analysis_file)
    
    # Create third plot: Revenue vs Solution Time (tradeoff plot)
    fig3 = plt.figure(figsize=(10, 6))
    for n_scenarios in sample_sizes:
        x_values = [x[2] for x in results[n_scenarios]]  # Solution times
        y_values = [x[1] for x in results[n_scenarios]]  # Revenues
        
        # Add annotations for the number of price points
        plt.plot(x_values, y_values, marker='o', label=f"{n_scenarios} scenarios")
        
        # Annotate each point with its price points
        for i, (price_points, revenue, solve_time) in enumerate(results[n_scenarios]):
            price_label = str(price_points) if price_points is not None else "Unl."
            plt.annotate(price_label, (x_values[i], y_values[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    # Add labels and legend
    plt.xlabel("Solution Time (seconds)")
    plt.ylabel("Expected Revenue ($)")
    plt.title("Revenue vs. Solution Time Tradeoff")
    plt.legend()
    plt.grid(True)
    
    # Save the third figure
    tradeoff_analysis_file = f"results/revenue_time_tradeoff_{timestamp}.png"
    plt.savefig(tradeoff_analysis_file)
    
    # Generate a summary table
    with open(f"results/price_points_analysis_{timestamp}.txt", 'w') as f:
        f.write("Price Points Analysis Results\n")
        f.write("==========================\n\n")
        
        # Table header
        f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        # Table rows for revenue
        for i, max_bid_prices in enumerate(max_bid_prices_list):
            price_label = str(max_bid_prices) if max_bid_prices is not None else "unlimited"
            f.write(f"{price_label:<15}")
            for n_scenarios in sample_sizes:
                revenue = results[n_scenarios][i][1]
                f.write(f"${revenue:>23.2f}")
            f.write("\n")
        
        # Add revenue improvement percentage vs 5 price points baseline
        f.write("\nRevenue Improvement % (vs. 5 price points):\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        # Calculate improvement percentages
        for i, max_bid_prices in enumerate(max_bid_prices_list):
            if i == 0:  # Skip the baseline (5 price points)
                price_label = str(max_bid_prices) if max_bid_prices is not None else "unlimited"
                f.write(f"{price_label:<15}")
                for n_scenarios in sample_sizes:
                    f.write(f"{'baseline':>23}")
                f.write("\n")
            else:
                price_label = str(max_bid_prices) if max_bid_prices is not None else "unlimited"
                f.write(f"{price_label:<15}")
                for n_scenarios in sample_sizes:
                    baseline_revenue = results[n_scenarios][0][1]
                    current_revenue = results[n_scenarios][i][1]
                    improvement_pct = (current_revenue - baseline_revenue) / baseline_revenue * 100
                    f.write(f"{improvement_pct:>23.2f}%")
                f.write("\n")
        
        # Add solve times
        f.write("\nSolve Times (seconds):\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        for i, max_bid_prices in enumerate(max_bid_prices_list):
            price_label = str(max_bid_prices) if max_bid_prices is not None else "unlimited"
            f.write(f"{price_label:<15}")
            for n_scenarios in sample_sizes:
                solve_time = results[n_scenarios][i][2]
                f.write(f"{solve_time:>23.2f}")
            f.write("\n")
        
        # Add solve time increase ratio vs 5 price points
        f.write("\nSolve Time Ratio (vs. 5 price points):\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        f.write(f"{'Price Points':<15}")
        for n_scenarios in sample_sizes:
            f.write(f"{n_scenarios:>15} scenarios")
        f.write("\n")
        f.write("-" * (15 + 15 * len(sample_sizes)) + "\n")
        
        for i, max_bid_prices in enumerate(max_bid_prices_list):
            if i == 0:  # Skip the baseline (5 price points)
                price_label = str(max_bid_prices) if max_bid_prices is not None else "unlimited"
                f.write(f"{price_label:<15}")
                for n_scenarios in sample_sizes:
                    f.write(f"{'baseline':>23}")
                f.write("\n")
            else:
                price_label = str(max_bid_prices) if max_bid_prices is not None else "unlimited"
                f.write(f"{price_label:<15}")
                for n_scenarios in sample_sizes:
                    baseline_time = results[n_scenarios][0][2]
                    current_time = results[n_scenarios][i][2]
                    time_ratio = current_time / baseline_time
                    f.write(f"{time_ratio:>23.2f}x")
                f.write("\n")
    
    print(f"\nAnalysis complete.")
    print(f"- Revenue vs Price Points plot: {revenue_analysis_file}")
    print(f"- Solution Time vs Price Points plot: {time_analysis_file}")
    print(f"- Revenue vs Solution Time tradeoff plot: {tradeoff_analysis_file}")
    print(f"- Detailed results table: results/price_points_analysis_{timestamp}.txt")
    
    return results

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run stochastic convergence bidding analysis")
    parser.add_argument("--analysis", action="store_true", help="Run price points analysis")
    parser.add_argument("--scenarios", type=int, default=100, help="Number of scenarios for main run")
    args = parser.parse_args()
    
    if args.analysis:
        # Run the analysis of price points vs. revenue
        results = run_bid_price_analysis()
    else:
        # Run the standard optimization to get detailed output
        revenue, solve_time = run_main_optimization(args.scenarios)
        print(f"Results saved to {output_file}")