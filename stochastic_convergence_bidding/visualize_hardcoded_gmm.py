import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import os

# Load the original data
try:
    original_archive = np.load('./data/energy_market_samples.npz')
    original_data = original_archive['data']
    
    # Load the hardcoded GMM-generated data
    gmm_archive = np.load('./data/hardcoded_gmm_samples.npz')
    gmm_data = gmm_archive['data']
    
    os.makedirs('./data/visualizations', exist_ok=True)
    
    # Select specific hours to visualize
    key_hours = [0, 3, 10, 16, 18, 20, 23]
    
    # Create a figure with subplots for each hour
    fig, axes = plt.subplots(len(key_hours), 2, figsize=(15, 4*len(key_hours)))
    
    for i, hour in enumerate(key_hours):
        # Original data for this hour
        orig_da = original_data[:, hour, 0]
        orig_rt = original_data[:, hour, 1]
        
        # GMM-generated data for this hour
        gmm_da = gmm_data[:, hour, 0]
        gmm_rt = gmm_data[:, hour, 1]
        
        # Calculate percentiles for better plotting
        da_min = min(np.percentile(orig_da, 1), np.percentile(gmm_da, 1))
        da_max = max(np.percentile(orig_da, 99), np.percentile(gmm_da, 99))
        rt_min = min(np.percentile(orig_rt, 1), np.percentile(gmm_rt, 1))
        rt_max = max(np.percentile(orig_rt, 99), np.percentile(gmm_rt, 99))
        
        # Create consistent bins for both plots
        da_bins = np.linspace(da_min, da_max, 50)
        rt_bins = np.linspace(rt_min, rt_max, 50)
        
        # Plot original data
        axes[i, 0].hist2d(
            orig_da, orig_rt, 
            bins=[da_bins, rt_bins],
            cmap='viridis', 
            norm=LogNorm()
        )
        axes[i, 0].set_title(f"Original Data - Hour {hour}")
        axes[i, 0].set_xlabel("DA Price ($/MWh)")
        axes[i, 0].set_ylabel("RT Price ($/MWh)")
        
        # Add 45-degree line
        min_val = min(da_min, rt_min)
        max_val = max(da_max, rt_max)
        axes[i, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Plot GMM-generated data
        axes[i, 1].hist2d(
            gmm_da, gmm_rt, 
            bins=[da_bins, rt_bins],
            cmap='viridis', 
            norm=LogNorm()
        )
        axes[i, 1].set_title(f"Hardcoded GMM Data - Hour {hour}")
        axes[i, 1].set_xlabel("DA Price ($/MWh)")
        axes[i, 1].set_ylabel("RT Price ($/MWh)")
        
        # Add 45-degree line
        axes[i,
         1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('./data/visualizations/hardcoded_gmm_comparison.png', dpi=300)
    print("Visualization saved to ./data/visualizations/hardcoded_gmm_comparison.png")
    
    # Calculate and print summary statistics
    print("\nSummary Statistics Comparison:")
    print("-" * 50)
    print(f"{'Dataset':<10} {'Variable':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    
    for name, data in [("Original", original_data), ("GMM", gmm_data)]:
        for i, var_name in enumerate(['dalmp', 'rtlmp']):
            values = data[:, :, i].flatten()
            print(f"{name:<10} {var_name:<10} {values.mean():10.2f} {values.std():10.2f} "
                  f"{values.min():10.2f} {values.max():10.2f}")
            
    # Also calculate DART spread statistics
    print("\nDART Spread Statistics:")
    print("-" * 50)
    print(f"{'Dataset':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    
    for name, data in [("Original", original_data), ("GMM", gmm_data)]:
        dart_spread = data[:, :, 0] - data[:, :, 1]  # DA - RT for all scenarios and hours
        print(f"{name:<10} {dart_spread.mean():10.2f} {dart_spread.std():10.2f} "
              f"{dart_spread.min():10.2f} {dart_spread.max():10.2f}")
        
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure both original and GMM-generated data files exist.")