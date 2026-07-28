# Claim 1: VERIFIED

**Reviewer verdict:** `VERIFIED`. **Cumulative run passed:** `True`.

## Exact claim contract

```json
{
  "claim_id": 1,
  "paper_anchor": "Section 3.2, Algorithm 1 (#alg1)",
  "exact_statement": "SOSMC carries a weighted particle population through successive parameter targets using reweighting, resampling, mutation, and a particle gradient estimate.",
  "acceptance": [
    "Execute the named Algorithm 1 path in an official EBM setting.",
    "Trace particle identity and nonuniform weights across at least two outer iterations.",
    "Independently recompute the gradient estimate and fail on disagreement."
  ],
  "baseline_verdict": "BLOCKED"
}
```

## Raw numerical result inline

10,000 official-EBM particles; three traced iterations; maximum relative gradient discrepancy `9.30e-7`; continuity control failed.

## Source and quantifiers



[Full source audit](../../evidence/claim_1/source_audit.md) ·
[contract JSON](../../evidence/claim_1/claim_contract.json)

## Method, code, and command

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

and compares it to the gradient produced by the official loss graph. The
relative L2 tolerance is fixed at `1e-6`. The componentwise absolute tolerance
is the float32 accumulation bound
`max(1e-6, 64*eps_float32*max(1, ||g||_2))`; it prevents a large-magnitude
10,000-term gradient from failing on representational roundoff while the
scale-free relative check remains strict. The negative control replaces the
carried-population hash with a fresh-population marker and must fail
specifically on particle continuity.


Current checker:
[`sosmc_repro/claim1_checker.py`](../../code/sosmc_repro/claim1_checker.py).
The cumulative entrypoint is
[`sosmc_repro/run.py`](../../code/sosmc_repro/run.py).

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

Pinned environment:
[`pyproject.toml`](../../code/pyproject.toml) and
[`uv.lock`](../../code/uv.lock).

## Evidence and independent checks

[Raw output JSON](../../evidence/claim_1/raw_output.json) ·
[independent checker output](../../evidence/claim_1/independent_checker_output.json) ·
[negative-control output](../../evidence/claim_1/negative_control_output.json) ·
[runtime/provenance](../../evidence/claim_1/runtime.json)

Run `199652d8-ec32-4192-a79f-d76f5ea9a46f` used Git `990cb3d8afd53accb03a9e48f0c57e2842137785`, seed family rooted at `20260726`,
Hugging Face `cpu-upgrade`, actual cgroup quota **8 CPU cores**, no CUDA, and
total cumulative runtime **7,644.54 seconds**. The verifier exits nonzero if a
contracted assertion or negative control changes:

```bash
python verify_release.py --root .
```

## Limitations and deviations

# Limitations and deviations

The frozen root baseline has no faithful EBM execution. This child supersedes
that blocked result with a trace from the official checkpointed 2D EBM code.
Only the first three of 1,001 outer iterations are retained at tensor-hash
granularity to bound evidence size; the full tuning run still executes.

The paired ImpDiff/SOSMC reference draw is reused only after bitwise
reference-parameter and exact sampler-configuration checks. This avoids a second
identical 20,000-step initialization without changing the SOSMC particle
population or any outer update.
