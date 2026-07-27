# Method

Load the authors' public checkpoint and execute the official
`SOSMCULATuner` on CPU for all 12 reward/regularization cells. All cells use the
same exact cached initial particle tensor only after bitwise checkpoint-state
and sampler-configuration equality checks. The outer-loop random seed is reset
after initialization so proposal noise is shared across cells. This
paper-text route uses Appendix E.3's pure-ULA step size `5e-3`; the released
notebook's executed `3e-3` sweep is preserved as a separate interpretation.

Terminal evaluation uses the original non-Gaussian pretraining sampler for 512
steps from identical initial states and noise at three fixed seeds, 64 samples
per seed. Report paired reward differences and normal-approximation 95%
intervals over the 192 paired trajectories.

Train a small deterministic convolutional MNIST recognizer for three epochs and
require at least 97% held-out accuracy before its diagnostics are admissible.
Standardize its penultimate features using 1,000 balanced training digits and
measure each generated sample's nearest support distance. Compare tuned samples
with the pretrained EBM and with exact reward-maximizing constant/half-plane
images. The latter must fail for the intended non-digit reason.

The recognizer and support-distance thresholds are fixed in source before the
formal run. They are not selected from the tuned result.
