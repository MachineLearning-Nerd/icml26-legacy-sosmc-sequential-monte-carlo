# Claim 5 source audit

Source: `https://ar5iv.labs.arxiv.org/html/2601.22003`, retrieved
2026-07-26 with an explicit browser User-Agent. SHA-256:
`ec13f8ac302898767ba1aedeefd94742c126415bbdc3db27b23a8ae89e6c7128`.

The operative statement is Section 5.2 paragraph `S5.SS2.p4.7`. It compares
ImpDiff and SOSMC-ULA across datasets, half-plane rewards, and regularisation
strengths. It reports higher objective contours for SOSMC at small
`beta_KL`, comparable objective values at large `beta_KL`, and closer tracking
of the fresh expected reward by SOSMC's weighted particle rewards.

Numerical and algorithmic assumptions are in Appendix E.2.2. The benchmark
datasets are circles, two moons, and blobs (`A5.SS2.p1.3`). The methods start
from a frozen pretrained EBM, use a one-step unconstrained ULA transition,
10,000 particles, Adam with learning rate `2e-4`, no particle reinitialisation,
SOSMC resampling threshold `0.9`, adaptation threshold `0.95`, and half-plane
indicator rewards. The illustrative setting uses `beta_KL=0.25` and 1,001
outer iterations; the supplied notebook also defines `beta_KL=5` as its
large-regularisation endpoint.

The paper's prose is empirical rather than universally quantified: it says
higher objective contours occur "in cases where" `beta_KL` is small, not for
every dataset/reward combination. This node directly tests that finite
illustrative case on circles at `beta_KL=0.25`, then tests the stated
large-regularisation comparability at `beta_KL=5`. Separate two-moons and blobs
shards audit whether the direction generalizes. Broader reward orientations
and stochastic replication remain stated limitations even if this exact
finite contract passes.
