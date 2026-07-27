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
iteration, Adam at `1e-4`, initial ULA step size `3e-3` in the supplied sweep,
ESS thresholds `0.9` and `0.95`, and rewards for brightness, darkness, and
lower-half concentration. The supplied notebook's sweep uses
`beta_KL in {5,2,1,0.5}`.

The pretraining/evaluation transition adds Gaussian jitter with standard
deviation `0.005`, clamps pixels to `[-1,1]`, clips each energy-gradient
component to `[-0.03,0.03]`, and takes a gradient step scaled by `10`. The paper
notes that this is not a Gaussian transition and does not satisfy detailed
balance. Tuning instead uses unclipped, unjittered pure Gaussian ULA so the
SOSMC kernel-density ratios are defined.

The paper assesses absence of reward hacking visually. This reproduction adds a
machine-checkable diagnostic: a separately trained recognizer, standardized
feature-space support distance to real MNIST, and reward-maximizing non-digit
controls. These diagnostics operationalize rather than weaken the qualitative
source statement.
