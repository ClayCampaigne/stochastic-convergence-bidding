# TODO Items for Stochastic Convergence Bidding Project

## Code Cleanup
- Remove all backup files and references to them
- Clean up any remaining development artifacts

## Out-of-Sample Evaluation
- Implement out-of-sample evaluation of bid curves for comparability across sample sizes
- Generate separate test data set that isn't used in optimization
- Evaluate optimized bid curves from different sample sizes against the same test data
- Calculate and compare out-of-sample expected revenue for each scenario count
- Add visualization comparing in-sample vs out-of-sample performance

## Performance Improvements
- Further optimize the parallel processing implementation
- Add caching of intermediate results for larger sample sizes
- Consider GPU acceleration for the largest sample sizes

## Analysis Extensions
- Analyze convergence rates for different price point configurations
- Investigate the impact of GMM parameters on optimization results
- Study the relationship between price point density and bid curve quality