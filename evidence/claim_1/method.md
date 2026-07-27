# Method

The current child executes the authors' `SOSMCULARewardTuner` on a checkpointed
2D EBM with 10,000 particles and 1,001 outer iterations. It hashes the particle
population before and after three consecutive outer iterations, counts ULA
proposals, audits normalized candidate weights and ESS before resampling, and
checks that the next iteration receives the exact prior population. When a
node contains multiple beta values, the same reference draw may be reused more
than once, but only after every reference state and sampler configuration
passes the bitwise equality guard.

For two iterations, a separate checker reconstructs the parameter gradient as

`sum_i w_i ((r_i-E_w[r]) + beta*(delta_i-E_w[delta])) grad_theta E_theta(x_i)`

and compares it to the gradient produced by the official loss graph. Tolerances
are fixed at `1e-6` for relative L2 and maximum absolute error. The negative
control replaces the carried-population hash with a fresh-population marker and
must fail specifically on particle continuity.
