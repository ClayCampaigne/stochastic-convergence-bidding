import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'energy_market_samples.npz')
archive = np.load(data_path)
data = archive['data']
target_names = archive['target_names']

# Extract DA and RT prices (not residuals)
dalmp = data[:, :, 0]  # Day-ahead prices
rtlmp = data[:, :, 1]  # Real-time prices

# Create a grid of subplots for all 24 hours
fig = plt.figure(figsize=(20, 15))
gs = gridspec.GridSpec(4, 6, figure=fig)

# Set color limits based on percentiles to handle extreme values
da_vmin, da_vmax = np.percentile(dalmp, [1, 99])
rt_vmin, rt_vmax = np.percentile(rtlmp, [1, 99])

# Function to plot joint distribution
def plot_joint_distribution(ax, x, y, hour):
    # Remove extreme outliers for better visualization (beyond 99.5th percentile)
    mask = (x < np.percentile(x, 99.5)) & (y < np.percentile(y, 99.5))
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    # Create a 2D histogram
    h, xedges, yedges = np.histogram2d(x_filtered, y_filtered, bins=50)
    
    # Plot as a heatmap with log scale for better visibility
    im = ax.imshow(h.T, origin='lower', aspect='auto', 
                  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                  cmap='viridis', norm=LogNorm())
    
    # Compute the 45-degree line (where DA=RT)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    min_val = max(min(xlim[0], ylim[0]), -100)  # Avoid extremely negative values
    max_val = min(max(xlim[1], ylim[1]), 200)   # Avoid extremely high values
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='DA=RT')
    
    # Labels and title
    ax.set_title(f"Hour {hour}")
    
    # Return for colorbar
    return im

# Plot each hour
for hour in range(24):
    # Calculate row and column for this hour
    row = hour // 6
    col = hour % 6
    
    # Create subplot
    ax = fig.add_subplot(gs[row, col])
    
    # Plot joint distribution
    im = plot_joint_distribution(ax, dalmp[:, hour], rtlmp[:, hour], hour)
    
    # Only add axis labels for leftmost and bottom plots
    if col == 0:
        ax.set_ylabel("RT Price ($/MWh)")
    if row == 3:
        ax.set_xlabel("DA Price ($/MWh)")

# Add a single set of axis labels for the entire figure
fig.text(0.5, 0.02, 'Day-Ahead Price ($/MWh)', ha='center', fontsize=14)
fig.text(0.02, 0.5, 'Real-Time Price ($/MWh)', va='center', rotation='vertical', fontsize=14)

# Add a color bar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Count (log scale)')

# Add overall title
fig.suptitle('Joint Distribution of Day-Ahead and Real-Time Prices by Hour', fontsize=16)

# Adjust layout
plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.95])

# Save figure
plt.savefig('/Users/claywcampaigne/Documents/stochastic-convergence-bidding/data/price_joint_distributions.png', dpi=300)
print("Plot saved to data/price_joint_distributions.png")