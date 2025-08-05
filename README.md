# Stochastic Convergence Bidding Optimization

## Overview

In this project we implement the **Sample-PV** algorithm from §II.D of  
*Letif Mones, Sean Lovett, “A General Stochastic Optimization Framework for Convergence Bidding”*.  
The key insight is that we can **optimize both bid prices and quantities** simultaneously via a **linear program** by:

- Precomputing a **scenario-by-bid-price payoff matrix** of DART spread revenues; multiplying this matrix by the vector of bid volumes (at each candidate price) yields a **scenario revenue vector**.
- Treating the decision as a pair of (typically sparse) **volume vectors** at **candidate DA prices**—one for the **sell** side and one for the **buy** side. With many scenarios, the sampled DA prices form a fine **price grid**, approximating a continuum.

---

### Core discretization (one node, one hour of the day)

1. **Scenario sample of prices**

   $$\bigl(\pi^{\mathrm{DA}}_i,\;\pi^{\mathrm{RT}}_i\bigr)_{i=1}^N, \qquad S_i \;:=\; \pi^{\mathrm{DA}}_i - \pi^{\mathrm{RT}}_i \;\; \text{(DART spread)}.$$

2. **Candidate bid price grid** from the sampled DA prices

   $$\{p_j\}_{j=1}^J, \quad \text{e.g. } \{p_j\} = \text{unique}\bigl\{\pi^{\mathrm{DA}}_i\bigr\}_{i=1}^N.$$

> No strategy can be evaluated at finer price resolution than the data itself,  
> so we let the optimization choose **volumes** $w_j$ at these **discrete prices**.

---

### Moneyness (clearing) indicator

- **Sell** side clears when the bid price is **at or below** the DA price:

  $$M^{\mathrm{sell}}_{ij} \;=\; \mathbb{1}\!\bigl\{\,\pi^{\mathrm{DA}}_i \;\ge\; p_j\,\bigr\}.$$

- **Buy** side clears when the bid price is **at or above** the DA price:

  $$M^{\mathrm{buy}}_{ij} \;=\; \mathbb{1}\!\bigl\{\,\pi^{\mathrm{DA}}_i \;\le\; p_j\,\bigr\}.$$

---

### Payoff matrices (sell-side only shown)

Form the **sell payoff matrix** by scaling the indicator with the DART spread:

$$\Delta^{\mathrm{sell}}_{ij} \;=\; S_i \, M^{\mathrm{sell}}_{ij}.$$

Given the decision variable of **sell volumes** $w \in \mathbb{R}^J_{\ge 0}$, the **scenario-wise revenues** are

$$r^{\mathrm{sell}} \;=\; \Delta^{\mathrm{sell}}\, w^\text{sell} \;\in\; \mathbb{R}^N.$$

(Analogously, for buy: $\Delta^{\mathrm{buy}}_{ij} = (-S_i)\,M^{\mathrm{buy}}_{ij}$ and $r^{\mathrm{buy}} = \Delta^{\mathrm{buy}}\,w^\text{buy}$.)

---

### Tiny matrix example (3 scenarios)

$N = 3$ scenarios and $J = 3$ candidate bid prices  

$$
\Delta^{\mathrm{sell}}
=
\begin{bmatrix}
S_1\,M_{11}^{\mathrm{sell}} & S_1\,M_{12}^{\mathrm{sell}} & S_1\,M_{13}^{\mathrm{sell}} \\[4pt]
S_2\,M_{21}^{\mathrm{sell}} & S_2\,M_{22}^{\mathrm{sell}} & S_2\,M_{23}^{\mathrm{sell}} \\[4pt]
S_3\,M_{31}^{\mathrm{sell}} & S_3\,M_{32}^{\mathrm{sell}} & S_3\,M_{33}^{\mathrm{sell}}
\end{bmatrix},
\qquad
r^{\mathrm{sell}} \;=\; \Delta^{\mathrm{sell}}\,w^\text{sell}
$$

with  

$$
M_{ij}^{\mathrm{sell}} = \mathbb{1}\!\bigl\{\,\pi^{\mathrm{DA}}_i \ge p_j\,\bigr\},
\quad
S_i = \pi^{\mathrm{DA}}_i - \pi^{\mathrm{RT}}_i,
\quad
w^\text{sell} = (w^\text{sell}_1,\;w^\text{sell}_2,\;w^\text{sell}_3)^{\!\top}.
$$

The decision variables are **sell volumes** $w^\text{sell}_j$ and **buy volumes** $w^\text{buy}_j$ at each candidate price $p_j$. The vector $r^{\mathrm{sell}}$ gives scenario-wise revenues from selling incremental volumes. This formulation naturally accommodates risk constraints like CVaR.

## Implementation Details

- **Scenario Generation**: Uses Gaussian Mixture Models to generate realistic electricity market price scenarios
- **Multi-hour Optimization**: We optimize across 24 hours rather than multiple nodes (as in the original paper), though this is just a matter of interpretation
- **CVaR Risk Constraint**: Limits downside risk while maximizing expected revenue
- **Flexible Price Discretization**: Reduces computational complexity by selecting a subset of sampled prices as candidate bid prices, rather than using all unique prices

## Setup

This project uses [UV](https://github.com/astral-sh/uv) for Python package management. To get started:

1. Install UV if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/ClayCampaigne/stochastic-convergence-bidding.git
   cd stochastic-convergence-bidding
   uv sync
   ```

## Usage

### Basic Optimization

Run the standard optimization with 100 scenarios:

```bash
uv run run_project.py
```

### Price Points Analysis

Run an analysis of the tradeoff between the number of price points and expected revenue:

```bash
uv run run_project.py --analysis
```

You can also specify the number of scenarios for the standard run:

```bash
uv run run_project.py --scenarios 500
```

## Analysis Results

The price points analysis produces several visualizations and a detailed results table:

1. **Revenue vs. Price Points Plot**: Shows how expected revenue changes with different numbers of price points
2. **Solution Time vs. Price Points Plot**: Shows how solution time scales with the number of price points
3. **Revenue vs. Solution Time Tradeoff Plot**: Visualizes the tradeoff between revenue and solution time

All results are saved in the `results/` directory with timestamps.

## Results and Analysis

### Key Findings

- Sell bids are placed in hours with large positive DART spreads (and small standard deviation), while buy bids occur in negative-spread hours
- Increasing the number of price points generally leads to higher expected revenue but with exponentially increasing solution time
- There is a clear tradeoff point where additional price points yield diminishing returns

### Computational Challenges

The main challenge is that solve time scales exponentially with problem size due to the matrix formulation. At 500 scenarios, the solver takes about 180 seconds. At larger problem sizes it quickly starts to take several hours. Results are also sensitive to sample size because of low-probability, high-impact spikes in the DART spread. For example, the model's bid in hour 22 (HE 23) varies significantly: the DART spread is -19.33 with 500 samples but -5.64 with 1000 samples.

### Future Work

Natural extensions include:
- Adding hour-specific volume constraints to manage tail risk
- Investigating speedup techniques to handle larger scenario sets
- Incorporating non-stationarity considerations into the risk management framework

## Decomposition Approaches for Speedup

As we extend to multiple nodes, solution time will rapidly blow up, making decomposition techniques crucial. Here are three promising approaches:

### 1. Dualize the CVaR Constraint

Add one multiplier $\lambda \ge 0$ to the constraint:

$$\rho - t + \frac{1}{(1-\alpha)N}\sum_i z_i \le 0$$

For fixed $\lambda$, the problem decomposes cleanly by node/hour (and even by scenario). A subgradient update on $\lambda$ enforces the global risk cap.

### 2. Progressive Hedging (PH)

Solve each scenario separately, add quadratic "agreement" penalties, average, and repeat. This approach is particularly fast for convex stochastic LPs.

### 3. Benders / L-shaped Cuts

Keep $t$ (and perhaps total volumes) in a master problem; scenario subproblems generate feasibility/optimality cuts for CVaR.

Any of these approaches replaces one big monolithic solve with many small per-node/hour solves plus a light coordination loop.
