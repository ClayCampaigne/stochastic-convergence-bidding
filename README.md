# Stochastic Convergence Bidding Optimization

## Overview

This project implements a stochastic optimization approach for convergence bidding in electricity markets, using Gaussian Mixture Models (GMMs) to generate synthetic price data and a Sample-PV algorithm for optimization.

## Features

- **Synthetic Data Generation**: Uses Gaussian Mixture Models to create realistic electricity market price scenarios
- **Multiple Bid Support**: Allows for multiple bid/offer price points at each hour
- **Price Points Analysis**: Analyzes the tradeoff between the number of price points, solution time, and expected revenue

## Usage

### Basic Optimization

Run the standard optimization with 100 scenarios:

```bash
python run_project.py
```

### Price Points Analysis

Run an analysis of the tradeoff between the number of price points and expected revenue:

```bash
python run_project.py --analysis
```

You can also specify the number of scenarios for the standard run:

```bash
python run_project.py --scenarios 500
```

## Analysis Results

The price points analysis produces several visualizations and a detailed results table:

1. **Revenue vs. Price Points Plot**: Shows how expected revenue changes with different numbers of price points
2. **Solution Time vs. Price Points Plot**: Shows how solution time scales with the number of price points
3. **Revenue vs. Solution Time Tradeoff Plot**: Visualizes the tradeoff between revenue and solution time

All results are saved in the `results/` directory with timestamps.

## Implementation Details

The project uses a custom MarketData class that can limit the number of price points used in the optimization. For the unlimited case, it uses all unique prices in the dataset. For limited cases, it evenly distributes the price points between the minimum and maximum prices for each hour.

The bidding model has been modified to retain all non-zero bids rather than just keeping the largest-volume bid per hour, allowing for more sophisticated bidding strategies.

## Key Findings

- Increasing the number of price points generally leads to higher expected revenue as it provides more flexibility in bidding
- Solution time increases exponentially with the number of price points, especially for larger scenario sizes
- There is a clear tradeoff point where additional price points yield diminishing returns in terms of revenue improvement but continue to increase solution time significantly