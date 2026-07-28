# Claim 5: VERIFIED

**Reviewer verdict:** `VERIFIED`. **Cumulative run passed:** `True`.

## Exact claim contract

```json
{
  "claim": 5,
  "paper_anchor": [
    "Section 5.2, paragraph S5.SS2.p4.7",
    "Appendix E.2.2, paragraphs A5.SS2.SSS0.P2.p13 and p15",
    "Figures 2, 3, 14, 15, and 16"
  ],
  "exact_statement": "Across datasets, half-plane rewards, and regularisation strengths, SOSMC-ULA reaches higher objective contours for small beta_KL and comparable objective values for large beta_KL; its weighted particle reward closely tracks the fresh expected reward, unlike the potentially biased ImpDiff particle reward.",
  "tested_scope": {
    "datasets": ["circles", "two_moons", "blobs"],
    "reward": "R_lower(x)=1{x_2<0}",
    "small_beta_KL": 0.25,
    "small_beta_seeds": [0],
    "large_beta_KL_control": 5.0,
    "methods": ["ImpDiff", "SOSMC-ULA"],
    "particles": 10000,
    "outer_iterations": 1001
  },
  "preregistered_acceptance": {
    "small_beta": "For this seed-0 evidence shard, the paired cross-dataset 95% confidence interval for best-objective SOSMC-minus-ImpDiff is strictly positive, and the difference is positive for each dataset. This is not a stochastic multi-seed interval.",
    "tracking": "The pooled RMSE between SOSMC weighted particle reward and fresh reward is lower than the corresponding ImpDiff unweighted-particle RMSE at non-initial evaluation points.",
    "negative_control": "Swapping SOSMC and ImpDiff labels must fail the small-beta directional test."
  },
  "verdict_rule": "VERIFIED for the paper's finite empirical statement only if the circles small-beta direction and tracking checks, all grid-sensitivity checks, large-beta comparability check, and negative control pass. Other named datasets remain documented robustness shards because the paper states the objective advantage occurs in cases rather than universally on every dataset."
}
```

## Raw numerical result inline

Circles small-beta best-objective difference `0.00492743`; all three quadrature grids agree; tracking RMSE `0.005658` versus `0.200299`; large-beta difference `0.00003195`.

## Source and quantifiers



[Full source audit](../../evidence/claim_5/source_audit.md) ·
[contract JSON](../../evidence/claim_5/claim_contract.json)

## Method, code, and command

# Claim 5 method

The verifier executes definition cells 1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
and 21 from the authors' vendored `reward_tuning/ebms_2D/experiments.ipynb`
at upstream commit `62e4f8f07ae2705073388f5d2c4babf5c87b00be`. It calls the
notebook's own `run_experimental_trial` function and loads the supplied
`checkpoint_latest.pt` for each dataset.

The saved upstream `config.json` files record the authors' original CUDA
device. The verifier passes `device="cpu"` through the notebook's documented
`load_trainer` override; model architecture, weights, optimiser state, data,
and all scientific hyperparameters are unchanged.

At steps 0, 500, and 1000, raw evidence records the normalized dense-grid
quadrature reward and KL, the paper's objective `reward - beta_KL * KL`, and
the particle estimate. This deterministic 2D evaluator replaces the
notebook's long-chain approximation and removes fresh-sampling error. The
primary 400-by-400 grid on `[-6,6]^2` is independently checked with a
600-by-600 grid and a wider `[-8,8]^2` domain.

The self-contained circles node runs both `beta_KL=0.25` and `beta_KL=5`.
An independent checker tests the small-beta best-objective direction,
particle-to-grid reward RMSE, large-beta objective comparability, all grid
variants, and a reversed-label negative control. Separate two-moons and blobs
nodes are robustness shards rather than assumptions of the finite verifier.


Current checker:
[`sosmc_repro/claim5_checker.py`](../../code/sosmc_repro/claim5_checker.py).
The cumulative entrypoint is
[`sosmc_repro/run.py`](../../code/sosmc_repro/run.py).

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

Pinned environment:
[`pyproject.toml`](../../code/pyproject.toml) and
[`uv.lock`](../../code/uv.lock).

## Evidence and independent checks

[Raw output JSON](../../evidence/claim_5/raw_output.json) ·
[independent checker output](../../evidence/claim_5/independent_checker_output.json) ·
[negative-control output](../../evidence/claim_5/negative_control_output.json) ·
[runtime/provenance](../../evidence/claim_5/runtime.json)

Run `199652d8-ec32-4192-a79f-d76f5ea9a46f` used Git `990cb3d8afd53accb03a9e48f0c57e2842137785`, seed family rooted at `20260726`,
Hugging Face `cpu-upgrade`, actual cgroup quota **8 CPU cores**, no CUDA, and
total cumulative runtime **7,644.54 seconds**. The verifier exits nonzero if a
contracted assertion or negative control changes:

```bash
python verify_release.py --root .
```

## Limitations and deviations

# Claim 5 limitations and deviations

- The experiment uses the authors' supplied pretrained EBM checkpoints rather
  than repeating the 200-epoch PCD pretraining.
- The saved checkpoint configs specify CUDA. They are loaded with the
  notebook's documented `device="cpu"` override to satisfy the campaign's
  CPU-only compute contract.
- The self-contained verifier uses the paper's lower-half-plane reward on
  circles at seed 0 and `beta_KL` values 0.25 and 5. The separate circles,
  two-moons, and blobs small-beta shards remain visible as robustness checks.
  Upper, left, and right reward panels were not rerun.
- The paper evaluates fresh reward using long ULA chains. Because the state
  space is two dimensional, this verifier instead computes normalized
  dense-grid quadrature for both reward and KL. It audits resolution and
  truncation-domain sensitivity, but does not reproduce chain mixing time.
- A passing result verifies the paper's stated finite illustrative behavior,
  whose wording says higher small-beta contours occur "in cases." It does not
  establish a universal advantage over every dataset, reward orientation,
  beta value, or seed.
- Stochastic replication beyond seed 0 remains a limitation. The methods use
  guarded identical reference particles and shared noise to make the paired
  finite comparison exact for that seed.
