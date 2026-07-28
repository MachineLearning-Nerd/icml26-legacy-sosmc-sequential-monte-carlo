# Claim 3: VERIFIED

**Reviewer verdict:** `VERIFIED`. **Cumulative run passed:** `True`.

## Exact claim contract

```json
{
  "claim_id": 3,
  "paper_anchor": "Section 4.2, Equation 19",
  "source_correction": "The exact Gaussian exponential ESS identity is Equation 19. Propositions 2-4 give separate local chi-squared expansions.",
  "assumptions": [
    "pi_theta is Gaussian N(theta, Sigma) with common positive-definite Sigma.",
    "theta_k-theta_{k-1} = -gamma grad l(theta_{k-1}).",
    "The idealized ESS proxy is rho=N/(1+chi^2(pi_theta_k||pi_theta_{k-1}))."
  ],
  "conclusion": "rho_k(gamma)=N exp(-gamma^2 ||grad l(theta_{k-1})||^2_{Sigma^{-1}}).",
  "acceptance": [
    "Symbolically derive the equal-covariance Gaussian chi-squared identity.",
    "Independently estimate it in multiple dimensions and nonspherical covariances.",
    "Negative control must reject an unequal-covariance extension."
  ]
}
```

## Raw numerical result inline

Maximum multidimensional Monte Carlo relative error `0.004325`; unequal covariance rejects the formula extension.

## Source and quantifiers



[Full source audit](../../evidence/claim_3/source_audit.md) ·
[contract JSON](../../evidence/claim_3/claim_contract.json)

## Method, code, and command

# Method

The verifier symbolically integrates the squared likelihood ratio in whitened
one-dimensional coordinates, then invokes orthogonal factorization for arbitrary
dimension. A seeded Monte Carlo checker uses independently generated dense
positive-definite covariances in dimensions 1, 2, 5, and 11.



Current checker:
[`sosmc_repro/theory.py`](../../code/sosmc_repro/theory.py).
The cumulative entrypoint is
[`sosmc_repro/run.py`](../../code/sosmc_repro/run.py).

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

Pinned environment:
[`pyproject.toml`](../../code/pyproject.toml) and
[`uv.lock`](../../code/uv.lock).

## Evidence and independent checks

[Raw output JSON](../../evidence/claim_3/raw_output.json) ·
[independent checker output](../../evidence/claim_3/independent_checker_output.json) ·
[negative-control output](../../evidence/claim_3/negative_control_output.json) ·
[runtime/provenance](../../evidence/claim_3/runtime.json)

Run `199652d8-ec32-4192-a79f-d76f5ea9a46f` used Git `990cb3d8afd53accb03a9e48f0c57e2842137785`, seed family rooted at `20260726`,
Hugging Face `cpu-upgrade`, actual cgroup quota **8 CPU cores**, no CUDA, and
total cumulative runtime **7,644.54 seconds**. The verifier exits nonzero if a
contracted assertion or negative control changes:

```bash
python verify_release.py --root .
```

## Limitations and deviations

# Limitations and deviations

Equation (19) is an idealized ESS proxy. Finite self-normalized particle ESS is
random and need not equal the proxy exactly. The result is not extended beyond
the paper's equal-covariance Gaussian assumptions.
