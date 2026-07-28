# Method

Load the authors' public checkpoint and execute the official
`SOSMCULATuner` on CPU for all 12 reward/regularization cells. All cells use the
same exact cached initial particle tensor only after bitwise checkpoint-state
and sampler-configuration equality checks. The outer-loop random seed is reset
after initialization so proposal noise is shared across cells. This third
interpretation route uses the released notebook's executed pure-ULA
step size `3e-3`, darkness factor `1/2`, and unscaled half-plane reward. The
first route's paper-text `5e-3` result remains immutable.

Terminal evaluation uses the original non-Gaussian pretraining sampler for 512
steps from identical initial states and noise at three fixed seeds, 64 samples
per seed. Report paired reward differences and normal-approximation 95%
intervals over the 192 paired trajectories.

Train a small deterministic convolutional MNIST recognizer for three epochs and
require at least 97% held-out accuracy as a reported semantic diagnostic. Its
confidence and feature distances do not decide anti-hacking acceptance because
Route 1 showed that they did not reject obvious constant-image controls.

The current acceptance/falsification test is a one-class digit-manifold
certificate. Fit a 64-component PCA subspace on the first 500 images of each
class in the official MNIST training set, with seed `2026072722`. For each
image, compute the mean standardized squared PCA coefficient and add its
orthogonal reconstruction MSE divided by the training-median residual. This
flags both off-subspace structure and extreme in-subspace coordinates such as
blank or saturated images. Fix the digit-manifold score limit to the larger of
the held-out-real q99 and pretrained EBM q95. Tuned samples and all controls are
excluded from fitting and threshold selection.

Verification requires every tuned cell's q95 anomaly score to remain below the
limit. Falsification requires an exact reward-improving cell whose conservative
95% order-statistic interval for the population median score lies wholly
above the limit. The pretrained baseline must remain accepted. The independent
controls are (i) the exact constant or half-plane reward maximizer and (ii)
held-out real MNIST images transformed by a fixed pixel permutation not used
in fitting. Both must be rejected.

The PCA rank, training count, seed, score definition, interval, and threshold
are fixed in source before this formal route. They are not selected from this
route's tuned result. Route 2's pixel morphology remains reported as a
diagnostic but cannot decide Route 3.

Compute estimate before launch: the cumulative fixed command is expected to
need 8 CPU cores and 1.5--2.5 hours, based on the earlier 6,162- and
8,092-second cumulative runtimes. It is therefore assigned to Hugging Face
`cpu-upgrade`; the PCA fit is expected to add less than five minutes within
that allocation. The run must report the actual CPU allocation, CUDA
visibility, and wall-clock runtime.
