import time
import numpy as np
import sys
from datetime import datetime

from stochastic_convergence_bidding.bidding_model import BiddingModel
from stochastic_convergence_bidding.market_data import MarketData
from stochastic_convergence_bidding.sample_data_generator import generate_sample_data

np.random.seed(1)

# Use multiple sample sizes
sample_sizes = [100]
hours = list(range(24))

# Create output file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"results_{timestamp}.txt"

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
    
    # Loop through each sample size
    for n_scenarios in sample_sizes:
        print(f"\n\n{'=' * 40}")
        print(f"Running optimization for {n_scenarios} scenarios...")
        print(f"{'=' * 40}\n")
    
        start_time = time.time()
    
        # Generate sample data
        data, target_names = generate_sample_data(
            num_samples=n_scenarios, num_hours=max(hours) + 1, random_seed=0
        )
        market_data = MarketData(data, target_names)
        
        # Print hourly statistics report
        market_data.print_hourly_report()
        
        # Flush output to ensure statistics are saved
        sys.stdout.flush()
    
        # Create and solve the Bidding Model
        model = BiddingModel(
            market_data=market_data,
            hours=hours,
            alpha=0.95,
            rho=-1000.0,
            verbose=False
        )
    
        model.build_model()
        model.solve_model()
        model.postprocess_bids()  # process and store all non-zero bids per hour
        objective_value = model.objective_value
    
        end_time = time.time()
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        print(f"Expected Revenue: ${objective_value:.2f}")
        
        # Flush output to ensure results are saved
        sys.stdout.flush()
    
        # Print all bids for each hour
        print("\nAll bids for each hour:")
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
            
        # Flush at the end of each sample size iteration
        sys.stdout.flush()
        
        # Write a clear separator to mark completion of this sample size
        print(f"\n{'=' * 40}")
        print(f"COMPLETED: {n_scenarios} scenarios")
        print(f"{'=' * 40}")
        sys.stdout.flush()
    
    # Restore original stdout
    sys.stdout = original_stdout

print(f"\nResults saved to {output_file}")