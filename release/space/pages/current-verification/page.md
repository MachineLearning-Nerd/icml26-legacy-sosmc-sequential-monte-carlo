# Current verification

This page supersedes the **Historical rejected baseline**. The current
cumulative verifier is [`verify_release.py`](../../verify_release.py), applied
to formal run `199652d8-ec32-4192-a79f-d76f5ea9a46f` at Git `990cb3d8afd53accb03a9e48f0c57e2842137785`.

## Result

| Claim | Verdict |
|---|---|
| 1 | VERIFIED |
| 2 | VERIFIED |
| 3 | VERIFIED |
| 4 | VERIFIED |
| 5 | VERIFIED |
| 6 | FALSIFIED |

## Visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | [Claim 1](../claim-1/page.md) | [source](../../code/sosmc_repro/) | yes | [JSON](../../evidence/claim_1/raw_output.json) | [checker](../../evidence/claim_1/independent_checker_output.json) | [control](../../evidence/claim_1/negative_control_output.json) | yes | VERIFIED |
| 2 | [Claim 2](../claim-2/page.md) | [source](../../code/sosmc_repro/) | yes | [JSON](../../evidence/claim_2/raw_output.json) | [checker](../../evidence/claim_2/independent_checker_output.json) | [control](../../evidence/claim_2/negative_control_output.json) | yes | VERIFIED |
| 3 | [Claim 3](../claim-3/page.md) | [source](../../code/sosmc_repro/) | yes | [JSON](../../evidence/claim_3/raw_output.json) | [checker](../../evidence/claim_3/independent_checker_output.json) | [control](../../evidence/claim_3/negative_control_output.json) | yes | VERIFIED |
| 4 | [Claim 4](../claim-4/page.md) | [source](../../code/sosmc_repro/) | yes | [JSON](../../evidence/claim_4/raw_output.json) | [checker](../../evidence/claim_4/independent_checker_output.json) | [control](../../evidence/claim_4/negative_control_output.json) | yes | VERIFIED |
| 5 | [Claim 5](../claim-5/page.md) | [source](../../code/sosmc_repro/) | yes | [JSON](../../evidence/claim_5/raw_output.json) | [checker](../../evidence/claim_5/independent_checker_output.json) | [control](../../evidence/claim_5/negative_control_output.json) | yes | VERIFIED |
| 6 | [Claim 6](../claim-6/page.md) | [source](../../code/sosmc_repro/) | yes | [JSON](../../evidence/claim_6/raw_output.json) | [checker](../../evidence/claim_6/independent_checker_output.json) | [control](../../evidence/claim_6/negative_control_output.json) | yes | FALSIFIED |

## Reproduce and verify

Formal full reproduction:

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

Fast independent evidence verification:

```bash
python verify_release.py --root .
```

The verifier returns zero only for the six displayed verdicts and exits
nonzero if an accepted check, evidence value, or intended negative-control
failure changes.

## Provenance

- Run: `199652d8-ec32-4192-a79f-d76f5ea9a46f`
- Git: `990cb3d8afd53accb03a9e48f0c57e2842137785`
- Environment: Python 3.12.13, exact [`uv.lock`](../../code/uv.lock)
- Estimate before run: more than one core and uncertain runtime, so HF
  `cpu-upgrade` was required.
- Actual allocation: 8 cgroup-quota CPU cores (64 host affinity), CUDA absent.
- Runtime: 7,644.5407 seconds; Claim 6: 4,992.9965 seconds.
- Seed root: 20260726; Claim 6 PCA seed: 2026072722.

See each claim page for assumptions, exact numerical evidence, raw downloads,
controls, limitations, and code.
