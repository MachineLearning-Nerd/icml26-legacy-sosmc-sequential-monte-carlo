# Claim 4: VERIFIED

**Reviewer verdict:** `VERIFIED`. **Cumulative run passed:** `True`.

## Exact claim contract

```json
{
  "claim_id": 4,
  "paper_anchor": "Section 5.1, Figure 1, Appendix E.1, Table 2",
  "statement_scope": "The four reported Gaussian-mixture Langevin/reward settings under Adam, each evaluated across 10 runs at the paper's beta and a synchronized two-second wall-clock budget.",
  "methods": ["ImpDiff", "SOUL", "SOSMC-ULA"],
  "acceptance": [
    "Mean terminal SOSMC-ULA reward exceeds ImpDiff in all four Table 2 settings.",
    "In Figure 1a's dual/smooth setting, SOSMC-ULA run-to-run standard deviation is below SOUL.",
    "In Figure 1b's tight/tight setting, SOUL exhibits the reported mode-transition failure while SOSMC-ULA exceeds it by at least 0.05 reward.",
    "The reversed-label checker must fail."
  ],
  "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "wall_clock_seconds": 2.001
}
```

## Raw numerical result inline

Ten seeds in each of four Table 2 settings; SOSMC mean exceeds ImpDiff in all four; reversed labels fail.

## Source and quantifiers



[Full source audit](../../evidence/claim_4/source_audit.md) ·
[contract JSON](../../evidence/claim_4/claim_contract.json)

## Method, code, and command

# Method

The runner executes the public authors' notebook cells directly from the
byte-for-byte vendored notebook at upstream commit `62e4f8f`. It does not
reimplement the three algorithms. For each of the four Table 2 settings it uses
the paper's beta, Adam configuration, 10 seeds, 10,000 particles/kernel
applications, warm-up step, and 2.001-second synchronized budget.

The raw evidence records every terminal reward, step count, and last recorded
time. A separate checker module recomputes means, sample standard deviations,
95% t intervals, paired SOSMC-minus-ImpDiff differences, and the preregistered
directional contract.



Current checker:
[`sosmc_repro/claim4_checker.py`](../../code/sosmc_repro/claim4_checker.py).
The cumulative entrypoint is
[`sosmc_repro/run.py`](../../code/sosmc_repro/run.py).

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

Pinned environment:
[`pyproject.toml`](../../code/pyproject.toml) and
[`uv.lock`](../../code/uv.lock).

## Evidence and independent checks

[Raw output JSON](../../evidence/claim_4/raw_output.json) ·
[independent checker output](../../evidence/claim_4/independent_checker_output.json) ·
[negative-control output](../../evidence/claim_4/negative_control_output.json) ·
[runtime/provenance](../../evidence/claim_4/runtime.json)

Run `199652d8-ec32-4192-a79f-d76f5ea9a46f` used Git `990cb3d8afd53accb03a9e48f0c57e2842137785`, seed family rooted at `20260726`,
Hugging Face `cpu-upgrade`, actual cgroup quota **8 CPU cores**, no CUDA, and
total cumulative runtime **7,644.54 seconds**. The verifier exits nonzero if a
contracted assertion or negative control changes:

```bash
python verify_release.py --root .
```

## Limitations and deviations

# Limitations and deviations

- CPU-only execution changes the number of outer steps completed in two seconds
  relative to the paper's personal computer/Colab hardware.
- Time-budget mode is inherently not bitwise deterministic even with fixed
  seeds; the official notebook explicitly documents this. A sibling experiment
  performs a deterministic step-matched calibration.
- This branch tests Adam because Table 2 reports both optimizers and the headline
  Figure 1b uses Adam; the cumulative campaign retains the precise scope.
