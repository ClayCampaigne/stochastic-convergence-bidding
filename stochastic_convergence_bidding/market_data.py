#!/usr/bin/env python3
"""
Data handling module for stochastic convergence bidding optimization.

This module provides utilities for:
- Converting raw data into an xarray.Dataset
- Storing and retrieving scenario-based energy market data
- Printing summary statistics, such as DART spread by hour.

Expected target_names: ['dalmp', 'rtlmp']
"""

from typing import Dict, List, Optional, cast

import numpy as np
import pandas as pd
import xarray as xr


class MarketData:
    """
    Encapsulates a scenario-based xarray.dataset of energy market information, including:
      - Day-ahead LMP (dalmp)
      - Real-time LMP (rtlmp)
    The dataset is indexed by (scenario, hour).

    Parameters:
    -----------
    numpy_data : np.ndarray
        3D numpy array of shape (num_samples, num_hours, num_targets).
        Typically, the third dimension includes ['dalmp', 'rtlmp'].
    target_names : List[str]
        The names corresponding to the third dimension in numpy_data.
        Expected to be exactly ['dalmp', 'rtlmp'] (in any order),
        or at least contain these two.
    dataset : Optional[xr.Dataset], default None
        If provided, we use this dataset directly instead of converting numpy_data.
        This can be used to load from a pre-existing xarray format.
    """
    def __init__(
        self,
        numpy_data: np.ndarray,
        target_names: List[str],
        dataset: Optional[xr.Dataset] = None,
        fixed_grid_prices: Optional[List[float]] = None,
    ):
        self.numpy_data = numpy_data
        self.target_names = target_names
        self.dataset: xr.Dataset = dataset if dataset is not None else xr.Dataset()
        self.fixed_grid_prices = fixed_grid_prices
        
        # If dataset not provided, build it from the numpy_data array
        if dataset is None and len(numpy_data.shape) == 3:
            num_samples, num_hours, num_targets = self.numpy_data.shape
            hours = np.arange(num_hours)
            scenarios = np.arange(num_samples)

            # Map each target name to a 2D array (scenario x hour)
            data_vars = {}
            for i, target in enumerate(self.target_names):
                data_vars[target] = (["scenario", "hour"], self.numpy_data[:, :, i])

            # Create the xarray Dataset
            self.dataset = xr.Dataset(
                data_vars=data_vars,
                coords={"scenario": scenarios, "hour": hours},
            )
    
    def get_xarray(self) -> xr.Dataset:
        """
        Return the underlying xarray.Dataset.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions: scenario x hour, containing data variables
            (e.g. dalmp, rtlmp, wind_power_mw).
        """
        return self.dataset
        
    def get_DA_bid_prices(self, hour: int, num_bid_prices: Optional[int] = None) -> np.ndarray:
        """
        Get bid prices for a given hour based on the specified strategy.
        
        Three possible behaviors:
        1. If self.fixed_grid_prices is set: use those fixed prices
        2. If num_bid_prices is provided: create evenly spaced prices between min/max observed prices
        3. Otherwise: use all unique observed DA prices (original behavior)

        Parameters
        ----------
        hour : int
            The hour index (0 <= hour < num_hours in the dataset).
        num_bid_prices : Optional[int], default None
            If provided, creates this many evenly spaced price points 
            between the min and max observed prices.

        Returns
        -------
        np.ndarray
            Array of price points to use for bidding in the specified hour.

        Raises
        ------
        Exception
            If the requested hour is out of range.
        """
        if hour > self.dataset.hour[-1]:
            raise Exception(
                f"Invalid hour: expected between 0 and {self.dataset.hour[-1].values}, got {hour}"
            )
            
        # Case 1: Use fixed grid prices if provided
        if self.fixed_grid_prices is not None:
            return np.array(self.fixed_grid_prices)
            
        # Get the hour data
        hour_data = self.dataset.sel(hour=hour)
        
        # Case 2: Create evenly spaced price points if num_bid_prices is specified
        if num_bid_prices is not None:
            min_price = np.min(hour_data["dalmp"].values)
            max_price = np.max(hour_data["dalmp"].values)
            return np.linspace(min_price, max_price, num_bid_prices)
            
        # Case 3: Default to original behavior - use all unique observed prices
        return np.unique(hour_data["dalmp"].values)
        
    def get_unique_DA_prices_for_hour(self, hour: int) -> np.ndarray:
        """
        Get the unique realized day-ahead (DA) clearing prices for a given hour.
        This is kept for backward compatibility.
        
        Parameters
        ----------
        hour : int
            The hour index (0 <= hour < num_hours in the dataset).

        Returns
        -------
        np.ndarray
            Array of unique DA prices (dalmp) for the specified hour.
        """
        # Simply call the new more flexible method with default parameters
        return self.get_DA_bid_prices(hour)

    def print_hourly_report(self) -> None:
        """
        Print a summary report for each hour, including:
          - average DART spread (dalmp - rtlmp),
          - standard deviation of DART spread.

        This method assumes that the dataset contains variables named:
          'dalmp', 'rtlmp'.
        """
        ds = self.dataset
        if not all(var in ds.data_vars for var in ["dalmp", "rtlmp"]):
            print("Missing one of the expected variables: dalmp, rtlmp.")
            return

        print("\nHourly Report:")
        hours = ds.hour.values
        for hour in hours:
            # For this hour, select scenario dimension
            dalmp_hour = ds.sel(hour=hour)["dalmp"]
            rtlmp_hour = ds.sel(hour=hour)["rtlmp"]

            # Compute DART spread
            dart_spread = dalmp_hour - rtlmp_hour
            avg_spread  = dart_spread.mean(dim="scenario").item()
            std_spread  = dart_spread.std(dim="scenario").item()

            print(
                f"Hour {hour:2d} | "
                f"Avg Spread = {avg_spread:7.2f}, "
                f"Std Spread = {std_spread:6.2f}"
            )