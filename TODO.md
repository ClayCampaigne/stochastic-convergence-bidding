# TODO Items for Stochastic Convergence Bidding Project

## Is there a bug? Larger sample sizes have lower revenue
- Does this pattern hold?
- Does it persist with risk constraint turned off?
- If it's a result of the risk constraint, do we need to adjust our analysis correspondingly? (what would this mean?)
- Implement out-of-sample evaluation. Does it persist out of sample?

## Code Cleanup
- Remove all backup files and references to them
- Clean up any remaining development artifacts
- review visualize_ and validate_gmm.py, and reference to hardcoded_gmm
- we can probably retain the original samples and the model fitting code

## Out-of-Sample Evaluation
- Use the same large sample for all OOS evals in a given run. See AI studio discussion
[//]: # (- Implement out-of-sample evaluation of bid curves for comparability across sample sizes)

[//]: # (- Generate separate test data set that isn't used in optimization)

[//]: # (- Evaluate optimized bid curves from different sample sizes against the same test data)

[//]: # (- Calculate and compare out-of-sample expected revenue for each scenario count)

[//]: # (- Add visualization comparing in-sample vs out-of-sample performance &#40;P3&#41;)

## Performance Improvements
- Use a decomposition method for the risk constraint

## Analysis Extensions
- Study the relationship between price point density and bid curve quality