# Method

Load the authors' public checkpoint and execute the official
`SOSMCULATuner` on CPU for all 12 reward/regularization cells. All cells use the
same exact cached initial particle tensor only after bitwise checkpoint-state
and sampler-configuration equality checks. The outer-loop random seed is reset
after initialization so proposal noise is shared across cells. This
second interpretation route uses the released notebook's executed pure-ULA
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

The new acceptance test uses classifier-independent image morphology. For each
sample it measures pixel standard deviation, horizontal/vertical total
variation, and nearest-neighbor distance to 1,000 balanced training digits
after fixed 4x4 average pooling to 7x7. The lower morphology limits retain 50%
of the smaller of the held-out-real and pretrained-baseline medians. The
support upper limit is the larger of held-out-real q95 and 1.5 times the
pretrained-baseline median. These calibrators exclude tuned samples and
controls. Every tuned cell must pass all three limits; exact
reward-maximizing constant/half-plane controls must fail at least one while
attaining at least the largest tuned reward.

The morphology definitions and thresholds are fixed in source before this
formal route. They are not selected from this route's tuned result.

Compute estimate before launch: the cumulative fixed command is expected to
need 8 CPU cores and 1.5--2.5 hours, based on Route 1's 6,162-second internal
runtime. It is therefore assigned to Hugging Face `cpu-upgrade`; the run must
report the actual CPU allocation, CUDA visibility, and wall-clock runtime.
