# Claim 6 source audit

Source: `https://ar5iv.labs.arxiv.org/html/2601.22003`, retrieved
2026-07-26 with an explicit browser User-Agent. SHA-256:
`ec13f8ac302898767ba1aedeefd94742c126415bbdc3db27b23a8ae89e6c7128`.

Section 5.3 says the convolutional MNIST EBM is pretrained with a sampling
procedure distinct from the ULA kernel used during tuning. It evaluates tuned
models with long chains under the original pretraining sampler. The operative
finite statement is that across all considered rewards and `beta_KL` values,
fresh reward increases relative to the pretrained baseline and samples remain
consistent with the regularization without reward hacking.

Appendix E.3 fixes the input domain at `28x28x1`, the convolutional Swish EBM
checkpoint, 1,000 particles, 1,000 outer iterations, one transition per outer
iteration, Adam at `1e-4`, an initial ULA step size reported as `5e-3`, ESS
thresholds `0.9` and `0.95`, and rewards for brightness, darkness, and
lower-half concentration. The supplied notebook's executed sweep instead uses
step size `3e-3` and `beta_KL in {5,2,1,0.5}`. Its saved textual outputs also
report `gamma=0.003000`. Route 1 followed the Appendix value `5e-3`. This
second route follows the authors' executed `3e-3` value so the disagreement is
tested rather than silently resolved.

The Appendix defines `R_dark=-mean(x)` and
`R_half=0.5*(mean(bottom)-mean(top))`. The executed darkness sweep locally
redefines `R_dark=-0.5*mean(x)`, while the notebook's half-plane helper omits
the factor `0.5`. Route 1 followed the written paper definitions; this second
route follows the saved executable scalings.

The pretraining/evaluation transition adds Gaussian jitter with standard
deviation `0.005`, clamps pixels to `[-1,1]`, clips each energy-gradient
component to `[-0.03,0.03]`, and takes a gradient step scaled by `10`. The paper
notes that this is not a Gaussian transition and does not satisfy detailed
balance. Tuning instead uses unclipped, unjittered pure Gaussian ULA so the
SOSMC kernel-density ratios are defined.

The paper assesses absence of reward hacking visually. Route 1 used a
separately trained recognizer and standardized feature-space support distance,
but rejected that diagnostic when all three visibly non-digit,
reward-maximizing controls fell inside its support threshold. This agrees with
the known possibility of high-confidence predictions for unrecognizable images
(Nguyen, Yosinski, and Clune, arXiv:1412.1897).

This second route follows the authors' executed `3e-3` sweep and executable
reward scalings. Its anti-hacking test is classifier-independent: pixel
standard deviation, image total variation, and nearest-neighbor distance in a
fixed 7x7 average-pooled pixel representation. Thresholds are calibrated only
from balanced held-out real MNIST and pretrained EBM samples. Tuned samples
and reward-maximizing controls are excluded from calibration.
