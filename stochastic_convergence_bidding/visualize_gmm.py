import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from sklearn.mixture import GaussianMixture
import os

def plot_hour_comparison(hour, original_data, gmm_data, save_path=None):
    """Plot comparison between original data and GMM-generated data for a specific hour."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get data for this hour
    orig_da = original_data[:, hour, 0]
    orig_rt = original_data[:, hour, 1]
    gmm_da = gmm_data[:, hour, 0]
    gmm_rt = gmm_data[:, hour, 1]
    
    # Original data 2D histogram
    h, xedges, yedges = np.histogram2d(
        orig_da, orig_rt, 
        bins=50, 
        range=[[np.percentile(orig_da, 1), np.percentile(orig_da, 99)], 
               [np.percentile(orig_rt, 1), np.percentile(orig_rt, 99)]]
    )
    axes[0].imshow(h.T, origin='lower', aspect='auto', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   cmap='viridis', norm=LogNorm())
    axes[0].set_title(f"Original Data - Hour {hour}")
    axes[0].set_xlabel("DA Price ($/MWh)")
    axes[0].set_ylabel("RT Price ($/MWh)")
    
    # Add 45-degree line (where DA=RT)
    xlim = axes[0].get_xlim()
    ylim = axes[0].get_ylim()
    min_val = max(min(xlim[0], ylim[0]), -100)
    max_val = min(max(xlim[1], ylim[1]), 200)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # GMM data 2D histogram
    h, xedges, yedges = np.histogram2d(
        gmm_da, gmm_rt, 
        bins=50,
        range=[[np.percentile(gmm_da, 1), np.percentile(gmm_da, 99)], 
               [np.percentile(gmm_rt, 1), np.percentile(gmm_rt, 99)]]
    )
    axes[1].imshow(h.T, origin='lower', aspect='auto', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                   cmap='viridis', norm=LogNorm())
    axes[1].set_title(f"GMM-Generated Data - Hour {hour}")
    axes[1].set_xlabel("DA Price ($/MWh)")
    
    # Add 45-degree line (where DA=RT)
    xlim = axes[1].get_xlim()
    ylim = axes[1].get_ylim()
    min_val = max(min(xlim[0], ylim[0]), -100)
    max_val = min(max(xlim[1], ylim[1]), 200)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    # Scatter plot of original vs GMM data (overlay)
    axes[2].scatter(orig_da, orig_rt, alpha=0.3, s=10, label='Original', color='blue')
    axes[2].scatter(gmm_da, gmm_rt, alpha=0.3, s=10, label='GMM', color='red')
    axes[2].set_title(f"Overlay Comparison - Hour {hour}")
    axes[2].set_xlabel("DA Price ($/MWh)")
    axes[2].set_ylabel("RT Price ($/MWh)")
    axes[2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
    axes[2].legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def main():
    try:
        # Load the original data
        original_archive = np.load('./data/energy_market_samples.npz')
        original_data = original_archive['data']
        
        # Load the GMM-generated data
        gmm_archive = np.load('./data/gmm_samples_final.npz')
        gmm_data = gmm_archive['data']
        
        os.makedirs('./data/visualizations', exist_ok=True)
        
        # Visualize key hours with interesting patterns
        key_hours = [
            0,   # Midnight
            3,   # Early morning
            8,   # Morning peak
            10,  # High volatility hour
            16,  # Afternoon
            18,  # Evening peak (high volatility)
            20,  # Evening (highest volatility)
            23,  # Late night
        ]
        
        for hour in key_hours:
            plot_hour_comparison(
                hour, 
                original_data, 
                gmm_data, 
                save_path=f'./data/visualizations/hour_{hour:02d}_comparison.png'
            )
        
        # Also create an overall summary visualization showing all hours
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(4, 6, figure=fig)
        
        # Calculate overall DART spread statistics for both datasets
        for i, (name, data) in enumerate([("Original", original_data), ("GMM", gmm_data)]):
            dart_spreads = data[:, :, 0] - data[:, :, 1]  # DA - RT for all scenarios and hours
            mean_spread = np.mean(dart_spreads)
            std_spread = np.std(dart_spreads)
            print(f"{name} Dataset - Mean DART spread: {mean_spread:.2f}, Std: {std_spread:.2f}")
        
        # Create a dense grid of all hour visualizations
        for hour in range(24):
            row = hour // 6
            col = hour % 6
            ax = fig.add_subplot(gs[row, col])
            
            orig_da = original_data[:, hour, 0]
            orig_rt = original_data[:, hour, 1]
            gmm_da = gmm_data[:, hour, 0]
            gmm_rt = gmm_data[:, hour, 1]
            
            # Plot original data as blue contours
            try:
                ax.scatter(orig_da, orig_rt, alpha=0.1, s=2, label='Original', color='blue')
                ax.scatter(gmm_da, gmm_rt, alpha=0.1, s=2, label='GMM', color='red')
            except Exception as e:
                print(f"Error plotting hour {hour}: {e}")
            
            # Add a diagonal line
            min_val = min(np.min(orig_da), np.min(gmm_da))
            max_val = max(np.max(orig_da), np.max(gmm_da))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            # Set title and optionally axis labels
            ax.set_title(f"Hour {hour}")
            
            # Only add axis labels for leftmost and bottom plots
            if col == 0:
                ax.set_ylabel("RT Price ($/MWh)")
            if row == 3:
                ax.set_xlabel("DA Price ($/MWh)")
        
        # Add a single legend for the entire figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)
        
        # Add overall title
        fig.suptitle('Comparison of Original vs. GMM-Generated Price Distributions by Hour', fontsize=16, y=0.99)
        
        # Adjust layout
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
        
        # Save the comparison figure
        plt.savefig('./data/visualizations/all_hours_comparison.png', dpi=300)
        print("Overall visualization saved to ./data/visualizations/all_hours_comparison.png")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure both original and GMM-generated data files exist.")

if __name__ == "__main__":
    main()