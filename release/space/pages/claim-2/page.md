# Claim 2: VERIFIED

**Reviewer verdict:** `VERIFIED`. **Cumulative run passed:** `True`.

## Exact claim contract

```json
{
  "claim_id": 2,
  "paper_anchor": "Section 4.1, Proposition 1 (#Thmproposition1), Equation 17",
  "source_correction": "The live judge calls this Proposition 2, but the retrieved paper labels it Proposition 1.",
  "assumptions": [
    "Z_theta is finite for every theta.",
    "A1: the loss is L-smooth and mu-Polyak-Lojasiewicz.",
    "The idealized SOSMC iterates use the exact distributions and reduce to exact gradient descent by Lemma 1.",
    "0 < gamma <= 1/L."
  ],
  "quantifier": "For every integer k >= 0.",
  "conclusion": "l(theta_k)-inf l <= (1-gamma*mu)^k (l(theta_0)-inf l).",
  "acceptance": [
    "Machine-check the symbolic one-step implication from smoothness, the step-size condition, and PL.",
    "Induct the one-step contraction to arbitrary k.",
    "Use exact arithmetic for a calibration grid independent of the claimed formula.",
    "Negative controls must break the certificate when gamma or PL is removed."
  ]
}
```

## Raw numerical result inline

Both symbolic residuals are exactly `0`; exhaustive integer grid has `0` violations; gamma/PL controls break the certificate.

## Source and quantifiers



[Full source audit](../../evidence/claim_2/source_audit.md) ·
[contract JSON](../../evidence/claim_2/claim_contract.json)

## Method, code, and command

# Method

The verifier constructs a proof certificate for the only nontrivial implication:
the smoothness descent bound at `gamma <= 1/L` yields a half-gradient decrease,
and PL substitutes `||grad l||^2 >= 2 mu (l-inf l)`. Iteration gives the stated
rate. SymPy checks the identities; an exact rational quadratic grid independently
calibrates the algebra without selecting instances from the target formula.



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

[Raw output JSON](../../evidence/claim_2/raw_output.json) ·
[independent checker output](../../evidence/claim_2/independent_checker_output.json) ·
[negative-control output](../../evidence/claim_2/negative_control_output.json) ·
[runtime/provenance](../../evidence/claim_2/runtime.json)

Run `199652d8-ec32-4192-a79f-d76f5ea9a46f` used Git `990cb3d8afd53accb03a9e48f0c57e2842137785`, seed family rooted at `20260726`,
Hugging Face `cpu-upgrade`, actual cgroup quota **8 CPU cores**, no CUDA, and
total cumulative runtime **7,644.54 seconds**. The verifier exits nonzero if a
contracted assertion or negative control changes:

```bash
python verify_release.py --root .
```

## Limitations and deviations

# Limitations and deviations

This verifies the idealized theorem, not a finite-particle convergence theorem.
The paper explicitly leaves finite-particle interacting-system theory to future
work in Remark 3. The certificate depends on Lemma 1's exact-gradient reduction.
