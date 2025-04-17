import numpy as np
import os
from typing import Tuple, List, Optional
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# GMM parameters for each hour
# Format: List of (weights, means, covariances) tuples for each hour's GMM
GMM_PARAMS = [
    # Hour 0 - 3 components
    {
        'weights': np.array([0.65, 0.25, 0.10]),
        'means': np.array([
            [42.8, 27.5],   # Main cluster
            [40.2, 40.2],   # Secondary cluster near DA=RT line
            [38.0, -10.5]   # Low RT price cluster
        ]),
        'covariances': np.array([
            [[25.0, 15.0], [15.0, 120.0]],   # Main cluster (positive correlation)
            [[20.0, 18.0], [18.0, 25.0]],    # Secondary cluster (strong correlation)
            [[15.0, -3.0], [-3.0, 80.0]]     # Low RT price cluster (weak negative correlation)
        ])
    },
    # Hour 1 - 3 components
    {
        'weights': np.array([0.7, 0.2, 0.1]),
        'means': np.array([
            [44.5, 25.0],   # Main cluster
            [42.0, 42.0],   # Secondary cluster near DA=RT line
            [40.0, -5.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[30.0, 18.0], [18.0, 140.0]],   # Main cluster
            [[18.0, 16.0], [16.0, 20.0]],    # Secondary cluster
            [[12.0, -2.0], [-2.0, 70.0]]     # Low RT price cluster
        ])
    },
    # Hour 2 - 3 components
    {
        'weights': np.array([0.7, 0.2, 0.1]),
        'means': np.array([
            [41.0, 18.5],   # Main cluster
            [38.0, 38.0],   # Secondary cluster
            [36.0, -10.0]   # Low RT price cluster
        ]),
        'covariances': np.array([
            [[28.0, 16.0], [16.0, 130.0]],  # Main cluster
            [[16.0, 14.0], [14.0, 18.0]],   # Secondary cluster
            [[10.0, -2.0], [-2.0, 65.0]]    # Low RT price cluster
        ])
    },
    # Hour 3 - 3 components with stronger peak
    {
        'weights': np.array([0.75, 0.15, 0.1]),
        'means': np.array([
            [38.0, 15.0],    # Main cluster
            [35.0, 35.0],    # Secondary cluster
            [33.0, -12.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[25.0, 14.0], [14.0, 120.0]],  # Main cluster
            [[15.0, 13.0], [13.0, 17.0]],   # Secondary cluster
            [[9.0, -1.5], [-1.5, 60.0]]     # Low RT price cluster
        ])
    },
    # Hour 4 - 3 components
    {
        'weights': np.array([0.75, 0.15, 0.1]),
        'means': np.array([
            [37.0, 15.0],   # Main cluster
            [34.0, 34.0],   # Secondary cluster
            [32.0, -10.0]   # Low RT price cluster
        ]),
        'covariances': np.array([
            [[22.0, 13.0], [13.0, 115.0]],  # Main cluster
            [[14.0, 12.0], [12.0, 16.0]],   # Secondary cluster
            [[8.0, -1.0], [-1.0, 55.0]]     # Low RT price cluster
        ])
    },
    # Hour 5 - 3 components
    {
        'weights': np.array([0.72, 0.18, 0.1]),
        'means': np.array([
            [39.0, 20.0],   # Main cluster
            [36.0, 36.0],   # Secondary cluster
            [34.0, -8.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[24.0, 14.0], [14.0, 120.0]],  # Main cluster
            [[15.0, 13.0], [13.0, 17.0]],   # Secondary cluster
            [[8.5, -1.0], [-1.0, 58.0]]     # Low RT price cluster
        ])
    },
    # Hour 6 - 3 components
    {
        'weights': np.array([0.68, 0.22, 0.1]),
        'means': np.array([
            [42.0, 30.0],   # Main cluster
            [39.0, 39.0],   # Secondary cluster
            [37.0, -5.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[26.0, 15.0], [15.0, 130.0]],  # Main cluster
            [[16.0, 14.0], [14.0, 18.0]],   # Secondary cluster
            [[9.0, -1.0], [-1.0, 60.0]]     # Low RT price cluster
        ])
    },
    # Hour 7 - 3 components
    {
        'weights': np.array([0.68, 0.22, 0.1]),
        'means': np.array([
            [48.0, 40.0],   # Main cluster
            [45.0, 45.0],   # Secondary cluster
            [43.0, 0.0]     # Low RT price cluster
        ]),
        'covariances': np.array([
            [[30.0, 18.0], [18.0, 150.0]],  # Main cluster
            [[18.0, 16.0], [16.0, 20.0]],   # Secondary cluster
            [[10.0, -1.0], [-1.0, 65.0]]    # Low RT price cluster
        ])
    },
    # Hour 8 - 3 components
    {
        'weights': np.array([0.65, 0.25, 0.1]),
        'means': np.array([
            [52.0, 45.0],   # Main cluster
            [49.0, 49.0],   # Secondary cluster
            [47.0, 10.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[35.0, 20.0], [20.0, 160.0]],  # Main cluster
            [[20.0, 18.0], [18.0, 22.0]],   # Secondary cluster
            [[12.0, -1.0], [-1.0, 70.0]]    # Low RT price cluster
        ])
    },
    # Hour 9 - 3 components
    {
        'weights': np.array([0.65, 0.25, 0.1]),
        'means': np.array([
            [52.0, 43.0],   # Main cluster
            [49.0, 49.0],   # Secondary cluster  
            [47.0, 10.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[35.0, 20.0], [20.0, 150.0]],  # Main cluster
            [[20.0, 18.0], [18.0, 22.0]],   # Secondary cluster
            [[12.0, -1.0], [-1.0, 70.0]]    # Low RT price cluster
        ])
    },
    # Hour 10 - 4 components (high volatility hour with price spikes)
    {
        'weights': np.array([0.60, 0.20, 0.15, 0.05]),
        'means': np.array([
            [45.0, 35.0],    # Main cluster
            [42.0, 42.0],    # Secondary cluster
            [40.0, 5.0],     # Low RT price cluster
            [43.0, 800.0]    # Extreme RT price spike
        ]),
        'covariances': np.array([
            [[30.0, 18.0], [18.0, 150.0]],    # Main cluster
            [[18.0, 16.0], [16.0, 20.0]],     # Secondary cluster
            [[10.0, -1.0], [-1.0, 65.0]],     # Low RT price cluster
            [[15.0, 50.0], [50.0, 150000.0]]  # Extreme spike (high variance)
        ])
    },
    # Hour 11 - 3 components
    {
        'weights': np.array([0.67, 0.23, 0.1]),
        'means': np.array([
            [44.0, 35.0],   # Main cluster
            [41.0, 41.0],   # Secondary cluster
            [39.0, 0.0]     # Low RT price cluster
        ]),
        'covariances': np.array([
            [[28.0, 17.0], [17.0, 140.0]],  # Main cluster
            [[17.0, 15.0], [15.0, 19.0]],   # Secondary cluster
            [[10.0, -1.0], [-1.0, 65.0]]    # Low RT price cluster
        ])
    },
    # Hour 12 - 3 components
    {
        'weights': np.array([0.67, 0.23, 0.1]),
        'means': np.array([
            [42.0, 26.0],   # Main cluster
            [39.0, 39.0],   # Secondary cluster
            [37.0, -5.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[26.0, 15.0], [15.0, 130.0]],  # Main cluster
            [[16.0, 14.0], [14.0, 18.0]],   # Secondary cluster
            [[9.0, -1.0], [-1.0, 60.0]]     # Low RT price cluster
        ])
    },
    # Hour 13 - 3 components
    {
        'weights': np.array([0.7, 0.2, 0.1]),
        'means': np.array([
            [38.0, 19.0],   # Main cluster
            [35.0, 35.0],   # Secondary cluster
            [33.0, -7.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[25.0, 15.0], [15.0, 125.0]],  # Main cluster
            [[15.0, 13.0], [13.0, 17.0]],   # Secondary cluster
            [[8.0, -1.0], [-1.0, 55.0]]     # Low RT price cluster
        ])
    },
    # Hour 14 - 3 components
    {
        'weights': np.array([0.7, 0.2, 0.1]),
        'means': np.array([
            [37.0, 18.0],   # Main cluster
            [34.0, 34.0],   # Secondary cluster
            [32.0, -8.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[24.0, 14.0], [14.0, 120.0]],  # Main cluster
            [[14.0, 12.0], [12.0, 16.0]],   # Secondary cluster
            [[8.0, -1.0], [-1.0, 55.0]]     # Low RT price cluster
        ])
    },
    # Hour 15 - 3 components with stronger bimodal pattern
    {
        'weights': np.array([0.4, 0.45, 0.15]),
        'means': np.array([
            [37.0, 19.0],   # First main cluster
            [34.0, -15.0],  # Second main cluster (negative RT)
            [36.0, 36.0]    # Near DA=RT line
        ]),
        'covariances': np.array([
            [[24.0, 14.0], [14.0, 120.0]],   # First main cluster
            [[20.0, -5.0], [-5.0, 80.0]],    # Second main cluster 
            [[15.0, 13.0], [13.0, 17.0]]     # Near DA=RT line
        ])
    },
    # Hour 16 - 4 components with strong bimodal pattern
    {
        'weights': np.array([0.35, 0.35, 0.2, 0.1]),
        'means': np.array([
            [38.0, 21.0],    # Upper main cluster
            [35.0, -25.0],   # Lower main cluster (negative RT)
            [36.0, 36.0],    # Near DA=RT line
            [-20.0, -15.0]   # Negative price cluster
        ]),
        'covariances': np.array([
            [[25.0, 15.0], [15.0, 140.0]],   # Upper main cluster
            [[25.0, -7.0], [-7.0, 100.0]],   # Lower main cluster
            [[16.0, 14.0], [14.0, 18.0]],    # Near DA=RT line
            [[10.0, 8.0], [8.0, 12.0]]       # Negative price cluster
        ])
    },
    # Hour 17 - 3 components
    {
        'weights': np.array([0.65, 0.25, 0.1]),
        'means': np.array([
            [41.0, 23.0],   # Main cluster
            [38.0, 38.0],   # Secondary cluster
            [36.0, -5.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[27.0, 16.0], [16.0, 135.0]],  # Main cluster
            [[16.0, 14.0], [14.0, 18.0]],   # Secondary cluster
            [[9.0, -1.0], [-1.0, 60.0]]     # Low RT price cluster
        ])
    },
    # Hour 18 - 4 components (volatile hour with price spikes)
    {
        'weights': np.array([0.60, 0.20, 0.15, 0.05]),
        'means': np.array([
            [50.0, 38.0],    # Main cluster
            [47.0, 47.0],    # Secondary cluster
            [45.0, 8.0],     # Low RT price cluster
            [48.0, 750.0]    # Extreme RT price spike
        ]),
        'covariances': np.array([
            [[32.0, 19.0], [19.0, 160.0]],    # Main cluster
            [[19.0, 17.0], [17.0, 21.0]],     # Secondary cluster
            [[11.0, -1.0], [-1.0, 70.0]],     # Low RT price cluster
            [[16.0, 60.0], [60.0, 140000.0]]  # Extreme spike (high variance)
        ])
    },
    # Hour 19 - 3 components
    {
        'weights': np.array([0.65, 0.25, 0.1]),
        'means': np.array([
            [53.0, 47.0],   # Main cluster
            [50.0, 50.0],   # Secondary cluster
            [48.0, 10.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[35.0, 21.0], [21.0, 170.0]],  # Main cluster
            [[20.0, 18.0], [18.0, 22.0]],   # Secondary cluster
            [[12.0, -1.0], [-1.0, 75.0]]    # Low RT price cluster
        ])
    },
    # Hour 20 - 4 components (most volatile hour with extreme price spikes)
    {
        'weights': np.array([0.55, 0.20, 0.15, 0.1]),
        'means': np.array([
            [52.0, 40.0],    # Main cluster
            [49.0, 49.0],    # Secondary cluster
            [47.0, 5.0],     # Low RT price cluster
            [50.0, 1000.0]   # Extreme RT price spike
        ]),
        'covariances': np.array([
            [[34.0, 20.0], [20.0, 170.0]],    # Main cluster
            [[20.0, 18.0], [18.0, 22.0]],     # Secondary cluster
            [[12.0, -1.0], [-1.0, 75.0]],     # Low RT price cluster
            [[18.0, 80.0], [80.0, 200000.0]]  # Extreme spike (very high variance)
        ])
    },
    # Hour 21 - 3 components
    {
        'weights': np.array([0.65, 0.25, 0.1]),
        'means': np.array([
            [48.0, 44.0],   # Main cluster
            [45.0, 45.0],   # Secondary cluster
            [43.0, 5.0]     # Low RT price cluster
        ]),
        'covariances': np.array([
            [[30.0, 18.0], [18.0, 150.0]],  # Main cluster
            [[18.0, 16.0], [16.0, 20.0]],   # Secondary cluster
            [[10.0, -1.0], [-1.0, 65.0]]    # Low RT price cluster
        ])
    },
    # Hour 22 - 3 components
    {
        'weights': np.array([0.68, 0.22, 0.1]),
        'means': np.array([
            [44.0, 38.0],   # Main cluster
            [41.0, 41.0],   # Secondary cluster
            [39.0, 0.0]     # Low RT price cluster
        ]),
        'covariances': np.array([
            [[28.0, 17.0], [17.0, 140.0]],  # Main cluster
            [[17.0, 15.0], [15.0, 19.0]],   # Secondary cluster
            [[10.0, -1.0], [-1.0, 65.0]]    # Low RT price cluster
        ])
    },
    # Hour 23 - 3 components
    {
        'weights': np.array([0.7, 0.2, 0.1]),
        'means': np.array([
            [40.0, 25.0],   # Main cluster
            [37.0, 37.0],   # Secondary cluster
            [35.0, -5.0]    # Low RT price cluster
        ]),
        'covariances': np.array([
            [[26.0, 15.0], [15.0, 130.0]],  # Main cluster
            [[16.0, 14.0], [14.0, 18.0]],   # Secondary cluster
            [[9.0, -1.0], [-1.0, 60.0]]     # Low RT price cluster
        ])
    }
]

def generate_sample_data(
    num_samples: int = 100,
    num_hours: int = 24,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate synthetic energy market data based on a Gaussian Mixture Model.
    
    Args:
        num_samples: Number of scenarios to generate.
        num_hours: Number of hours per scenario.
        random_seed: Optional seed for reproducibility.
        
    Returns:
        Tuple containing:
            - 3D numpy array of shape (num_samples, num_hours, 2) with DALMP and RTLMP
            - List of target names ['dalmp', 'rtlmp']
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Target names for the generated data
    target_names = ['dalmp', 'rtlmp']
    
    # Initialize output array
    output_data = np.zeros((num_samples, num_hours, len(target_names)))
    
    # Use pre-specified GMM parameters to generate data for each hour
    for hour in range(min(num_hours, len(GMM_PARAMS))):
        # Get the GMM parameters for this hour
        gmm_params = GMM_PARAMS[hour]
        
        # Create mixture samples directly using the precomputed GMM parameters
        # First, select which component each sample comes from
        component_indices = np.random.choice(
            len(gmm_params['weights']), 
            size=num_samples, 
            p=gmm_params['weights']
        )
        
        # Generate samples from each selected component
        samples = np.zeros((num_samples, 2))
        for i in range(num_samples):
            component = component_indices[i]
            samples[i] = np.random.multivariate_normal(
                mean=gmm_params['means'][component],
                cov=gmm_params['covariances'][component]
            )
        
        # Store the generated samples in the output array
        output_data[:, hour, 0] = samples[:, 0]  # DA prices
        output_data[:, hour, 1] = samples[:, 1]  # RT prices
        
    return output_data, target_names

def save_sample_data(
    data: np.ndarray,
    target_names: List[str],
    output_dir: str = "./data",
    filename: str = "energy_market_samples.npz",
) -> None:
    """
    Save generated data to a compressed numpy file.
    
    Args:
        data: The generated data array.
        target_names: Names of the variables.
        output_dir: Directory to save to.
        filename: Name of output file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path to output file
    output_path = os.path.join(output_dir, filename)
    
    # Save data and target names to compressed numpy file
    np.savez_compressed(output_path, data=data, target_names=target_names)
    
    # Compute some statistics for verification
    num_samples, num_hours, num_features = data.shape
    
    # Print statistics for verification
    print(f"Saved {num_samples} scenarios with {num_hours} hours each to {output_path}")
    print(f"Data shape: {data.shape}")
    print(f"Target names: {target_names}")
    
    # Print summary statistics for each feature
    for i, name in enumerate(target_names):
        feature_data = data[:, :, i].flatten()
        print(f"{name} - Mean: {feature_data.mean():.2f}, "
              f"Std: {feature_data.std():.2f}, "
              f"Min: {feature_data.min():.2f}, "
              f"Max: {feature_data.max():.2f}")

def load_sample_data(
    filepath: str = "./data/energy_market_samples.npz",
) -> Tuple[np.ndarray, List[str]]:
    """
    Load previously saved sample data.
    
    Args:
        filepath: Path to the data file.
        
    Returns:
        Tuple containing loaded data and target names.
    """
    # Load the compressed numpy file
    archive = np.load(filepath)
    
    # Extract data and target names
    data = archive['data']
    target_names = archive['target_names'].tolist()
    
    return data, target_names

def display_sample(
    data: np.ndarray, target_names: List[str], sample_idx: int = 0
) -> None:
    """
    Display a single scenario from the dataset.
    
    Args:
        data: The full dataset.
        target_names: Variable names.
        sample_idx: Which scenario to display.
    """
    # Check if the requested sample index is valid
    if sample_idx < 0 or sample_idx >= data.shape[0]:
        raise ValueError(f"Sample index {sample_idx} out of range. Dataset has {data.shape[0]} samples.")
    
    # Extract the selected sample
    sample_data = data[sample_idx]
    
    # Print header
    print(f"Sample {sample_idx} Data:")
    print("Hour", end="\t")
    for name in target_names:
        print(f"{name}", end="\t")
    print()
    
    # Print data for each hour
    for hour in range(sample_data.shape[0]):
        print(f"{hour}", end="\t")
        for feature_idx in range(len(target_names)):
            print(f"{sample_data[hour, feature_idx]:.2f}", end="\t")
        print()
    
    # Create a simple visualization of the sample
    plt.figure(figsize=(10, 6))
    
    # Plot each feature across all hours
    for i, name in enumerate(target_names):
        plt.plot(range(sample_data.shape[0]), sample_data[:, i], 
                 label=name, marker='o', linestyle='-')
    
    plt.title(f"Sample {sample_idx} - Energy Market Prices")
    plt.xlabel("Hour")
    plt.ylabel("Price ($/MWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()