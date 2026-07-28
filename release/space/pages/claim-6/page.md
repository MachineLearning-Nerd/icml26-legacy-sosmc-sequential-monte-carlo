# Claim 6: FALSIFIED

**Reviewer verdict:** `FALSIFIED`. **Cumulative run passed:** `True`.

## Exact claim contract

```json
{
  "claim_id": 6,
  "paper_anchor": [
    "Section 5.3",
    "Appendix E.3",
    "Figures 5 and 18"
  ],
  "exact_statement": "With an MNIST EBM pretrained using the clipped jitter-and-gradient sampler but tuned with pure Gaussian ULA, SOSMC-ULA increases fresh reward across the three stated rewards and beta_KL values while preserving digit-like structure rather than exploiting the reward.",
  "finite_domain": {
    "rewards": ["bright", "dark", "lower_half"],
    "beta_KL": [5.0, 2.0, 1.0, 0.5],
    "method": "SOSMC-ULA",
    "particles": 1000,
    "outer_iterations": 1000
  },
  "acceptance": [
    "Run all 12 reward/beta cells from the same checkpoint and guarded identical initial particles.",
    "For every cell, the paired 95% interval for terminal fresh reward minus pretrained fresh reward is strictly positive under the original pretraining sampler.",
    "A separately trained MNIST recognizer must achieve at least 97% test accuracy.",
    "Fit a deterministic 64-component PCA digit manifold on 5,000 balanced real MNIST training images. Define the anomaly score as mean standardized squared PCA coefficient plus reconstruction MSE divided by the training-median residual. Fix its limit at max(held-out-real q99, pretrained-baseline q95), excluding tuned samples and controls.",
    "A verification requires every tuned cell's anomaly-score q95 to remain at or below the fixed limit. A falsification requires at least one exact cell's 95% order-statistic interval for the median anomaly score to lie strictly above the limit while its paired fresh-reward interval is strictly positive.",
    "The exact reward-maximizing constant or half-plane image and an unseen fixed pixel-shuffling corruption of held-out real digits must lie above the anomaly-score limit, while the pretrained baseline remains below it.",
    "Swapping pretrained and tuned reward labels must fail the every-cell reward-improvement check."
  ],
  "verdict_rule": "VERIFIED only if every finite-grid, kernel, recognizer, digit-manifold, initialization, and negative-control check passes. FALSIFIED if all exact setup and control checks pass and at least one reward-improving finite-grid cell has its median PCA digit-manifold anomaly-score interval strictly beyond the independently fixed limit. Otherwise BLOCKED."
}
```

## Raw numerical result inline

All 12 reward intervals are positive. At fixed anomaly limit `4.170981`, bright/beta=.5 and lower-half/beta=1,.5 have median 95% intervals entirely above the limit.

## Source and quantifiers



[Full source audit](../../evidence/claim_6/source_audit.md) ·
[contract JSON](../../evidence/claim_6/claim_contract.json)

## Method, code, and command

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


Current checker:
[`sosmc_repro/claim6_checker.py`](../../code/sosmc_repro/claim6_checker.py).
The cumulative entrypoint is
[`sosmc_repro/run.py`](../../code/sosmc_repro/run.py).

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

Pinned environment:
[`pyproject.toml`](../../code/pyproject.toml) and
[`uv.lock`](../../code/uv.lock).

## Evidence and independent checks

[Raw output JSON](../../evidence/claim_6/raw_output.json) ·
[independent checker output](../../evidence/claim_6/independent_checker_output.json) ·
[negative-control output](../../evidence/claim_6/negative_control_output.json) ·
[runtime/provenance](../../evidence/claim_6/runtime.json)

Run `199652d8-ec32-4192-a79f-d76f5ea9a46f` used Git `990cb3d8afd53accb03a9e48f0c57e2842137785`, seed family rooted at `20260726`,
Hugging Face `cpu-upgrade`, actual cgroup quota **8 CPU cores**, no CUDA, and
total cumulative runtime **7,644.54 seconds**. The verifier exits nonzero if a
contracted assertion or negative control changes:

```bash
python verify_release.py --root .
```

## Limitations and deviations

# Limitations and deviations

The paper's “no reward hacking” conclusion is qualitative and image-based.
Route 1's learned recognizer-feature support criterion was rejected because it
failed all three obvious non-digit controls. Route 2 used simple,
classifier-independent pixel morphology and rejected five low-beta cells. This
was an independent quantitative operationalization, not a claim that standard
deviation, total variation, and pooled-pixel support fully capture human
perception, so Route 2 remained `BLOCKED`.

Route 3 uses a one-class linear digit manifold. Its PCA anomaly score is still
not identical to human judgment, and it can penalize valid brightness or
geometric changes. For that reason the falsification rule requires a
conservative population-median separation rather than one outlier, accepts the
pretrained EBM under the same threshold, and validates the detector on exact
maximizers and an unseen structure-destroying pixel permutation. Any
established counterexample will be reported only for the exact cell(s)
satisfying all conditions; it will not be generalized beyond the released
configuration.

The formal terminal evaluator disables the notebook's repeated 50-step fresh
evaluations during tuning. It retains the exact 1,000 optimization iterations
and uses a stronger terminal evaluation: three paired 64-sample, 512-step runs
under the original pretraining sampler. This avoids using intermediate
evaluation randomness to change optimization and directly tests the paper's
terminal robustness statement.

Appendix E.3 writes `R_dark=-mean(x)` and the lower-half reward with a factor
`1/2`. The executed darkness sweep introduces a factor `1/2`, while the
notebook half-plane helper omits it. Routes 2 and 3 follow the executable
notebook scalings. Route 1 followed the written formulas and remains preserved.
The optimizer, model, sampler, particle count, iteration count, and
regularization grid remain the authors' implementation.

Appendix E.3 reports an initial tuning step size of `5e-3`, whereas the released
notebook's beta sweep and saved outputs use `3e-3`. Route 1 followed the
paper-text value (`5e-3`). Routes 2 and 3 follow the authors' executed value
(`3e-3`). No recorded route is changed retrospectively after its run.
