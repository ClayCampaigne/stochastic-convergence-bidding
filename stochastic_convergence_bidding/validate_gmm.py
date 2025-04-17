import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
import os

# Load original data
try:
    original_archive = np.load('./data/energy_market_samples.npz')
    original_data = original_archive['data']
    original_target_names = original_archive['target_names']
    
    # Load GMM-generated data
    gmm_archive = np.load('./data/new_gmm_samples.npz')
    gmm_data = gmm_archive['data']
    gmm_target_names = gmm_archive['target_names']
    
    # Create a figure to compare distributions for a few representative hours
    representative_hours = [0, 8, 16, 23]  # Dawn, morning, afternoon, night
    fig, axes = plt.subplots(len(representative_hours), 2, figsize=(15, 15))
    
    for i, hour in enumerate(representative_hours):
        # Original data for this hour
        orig_da = original_data[:, hour, 0]
        orig_rt = original_data[:, hour, 1]
        
        # GMM data for this hour
        gmm_da = gmm_data[:, hour, 0]
        gmm_rt = gmm_data[:, hour, 1]
        
        # Plot original data
        ax = axes[i, 0]
        h, xedges, yedges = np.histogram2d(
            orig_da, orig_rt, 
            bins=50, 
            range=[[np.percentile(orig_da, 1), np.percentile(orig_da, 99)], 
                   [np.percentile(orig_rt, 1), np.percentile(orig_rt, 99)]]
        )
        im = ax.imshow(h.T, origin='lower', aspect='auto', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                      cmap='viridis', norm=LogNorm())
        ax.set_title(f"Original Data - Hour {hour}")
        ax.set_xlabel("DA Price ($/MWh)")
        ax.set_ylabel("RT Price ($/MWh)")
        
        # Add 45-degree line (where DA=RT)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_val = max(min(xlim[0], ylim[0]), -100)
        max_val = min(max(xlim[1], ylim[1]), 200)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        # Plot GMM-generated data
        ax = axes[i, 1]
        h, xedges, yedges = np.histogram2d(
            gmm_da, gmm_rt, 
            bins=50,
            range=[[np.percentile(gmm_da, 1), np.percentile(gmm_da, 99)], 
                   [np.percentile(gmm_rt, 1), np.percentile(gmm_rt, 99)]]
        )
        im = ax.imshow(h.T, origin='lower', aspect='auto', 
                      extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                      cmap='viridis', norm=LogNorm())
        ax.set_title(f"GMM-Generated Data - Hour {hour}")
        ax.set_xlabel("DA Price ($/MWh)")
        ax.set_ylabel("RT Price ($/MWh)")
        
        # Add 45-degree line (where DA=RT)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        min_val = max(min(xlim[0], ylim[0]), -100)
        max_val = min(max(xlim[1], ylim[1]), 200)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # Add overall title
    fig.suptitle('Comparison of Original vs. GMM-Generated Price Distributions', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the comparison figure
    plt.savefig('./data/gmm_validation.png', dpi=300)
    print("Validation plot saved to ./data/gmm_validation.png")
    
    # Print summary statistics for both datasets
    print("\nSummary Statistics Comparison:")
    print("-" * 50)
    print(f"{'Dataset':<10} {'Variable':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 50)
    
    for name, data in [("Original", original_data), ("GMM", gmm_data)]:
        for i, var_name in enumerate(['dalmp', 'rtlmp']):
            values = data[:, :, i].flatten()
            print(f"{name:<10} {var_name:<10} {values.mean():10.2f} {values.std():10.2f} "
                  f"{values.min():10.2f} {values.max():10.2f}")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure both original and GMM-generated data files exist.")