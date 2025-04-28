# Report
## Methods
We implement the Sample-PV algorithm described in sec II.D of
Letif Mones, Sean Lovett, A Generalized Stochastic Optimization Framework for Convergence Bidding.
(https://arxiv.org/pdf/2210.06543) 
However in our setting, we sum revenues across N hours, not N nodes. Other than this slight difference in
interpretation, the formulation is identical.

The advantage of this approach is that we can optimize both bid prices and quantities simultaneously with an LP 
formulation. Subject to the limitations that it does not guarantee a single bid and offer, it is a fully 
general approach that is suitable to any distribution, given enough samples.

Restricting attention for the moment to a single hour, the idea is that we place a (typically-zero)
bid volume at every DA price that is sampled. With a high number of samples, this approximates a
continuum of possible bid prices. We precompute a matrix of DART spread revenues such that when we
multiply the matrix by the vector of bid volumes at each potential price, we get a vector of DART
spread revenues, one for each scenario. This allows us to optimize both bid prices and quantities in a single LP.
We compute this matrix for each hour of the horizon, summing the revenues across hours.
We maximize expected revenue subject to a CVaR constraint.

In addition to the CVaR constraint, we place two constraints on the total bid volume summed across the day,
one on the offer volume and one on the bid volume. This allows us to manage risk while exploiting
different DART spread distributions in different hours.

We also validate our code with two behavioral tests: comparison with a lower bound strategy which self-schedules the 
same quantity across all scenarios, and an upper bound which cheats and chooses the optimal self-schedule bid separately
in each scenario.

We wrap the data in a class that contains an xarray dataset, to balance the efficiency of numpy with the
legibility of a labeled datastructure. Something like this may be preferable to e.g. pandas. 
Another data package to consider might by polars. 

## Results
The results of running run_project.py are shown in results.txt.
Examining the relationship between summary statistics of the DART spread and the bids and offers,
we see that offers are placed in hours with large positive spreads (and small standard deviation),
and bids in negative-spread hours.
This is consistent with the general idea of expected value maximization subject to a CVaR constraint.


## Challenges
One of the main challenges we face is that the solve time seems to scale exponentially with problem size,
because of the way the matrix formulation scales with sample size.
At 500 scenarios, the solver takes about 180 seconds.

Because of low probability high-impact spikes in the DART spread, the results are sensitive to sample size.
For example, the model places a large bid in hour 22 (HE 23). 
The DART spread is -19.33 with sample size 500, and -5.64 with sample size 1000. 
Given that our algorithm has scaling issues, we will need to experiment with sample size and see if we 
can get speedups so that the optimization results reflect the risk profile of the true distribution.

## Followup work:
Natural extensions would include adding more ad-hoc constraints: for example, limiting volume in individual 
hours to manage risk, especially where we are concerned that we may not be accurately capturing tail risk.
(See the issue above where the spread in HE 23 depends strongly on sample size.)
Adding ad-hoc risk-management constraints may be particularly important as the true data distribution
is probably nonstationary, so that any estimation method will have a hard time capturing tail risk.
Relatedly, we need to investigate how to speed up solution times.