Previous live judged score: `3/12`

Conservative projected score range after the proposed change: **9–12/12**

Best-supported possible new score: **12/12 (forecast, not a judge result)**

# Release report

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 | 1 | 2 | HIGH | VERIFIED | Official 2D EBM Algorithm 1 trace replaces the Gaussian toy; evaluator may interpret the framework claim differently. |
| 2 | 1 | 2 | HIGH | VERIFIED | Symbolic certificate covers the universal quantifier and exact assumptions; source numbering differs from the public paraphrase. |
| 3 | 1 | 2 | HIGH | VERIFIED | Symbolic equal-covariance Gaussian identity plus independent multidimensional Monte Carlo and an unequal-covariance control. |
| 4 | 0 | 2 | HIGH | VERIFIED | Direct four-setting, ten-seed, synchronized wall-clock run; sparse/hard paired CI includes zero although its paper-scope mean direction passes. |
| 5 | 0 | 2 | MEDIUM | VERIFIED | Direct circles illustrative case, three quadrature grids, tracking and large-beta control; not a complete multi-seed replication of every 2D dataset. |
| 6 | 0 | 2 | MEDIUM | FALSIFIED | Exact 12-cell sweep and calibrated one-class counterexample; digit-like structure remains an interpretation-sensitive perceptual concept. |

Current total score: **3/12** (live judge).

Conservative projected total score range: **9–12/12**.

Best-supported possible total score: **12/12**, strictly a forecast.

Claims changed in the candidate: Claims 1–3 replace toy evidence with direct or
proof-level evidence; Claims 4–5 are directly verified; Claim 6 is falsified
in three assumption-matched executable cells. No claim is BLOCKED in the
candidate. The exact publication action is a text-only incremental commit to
the existing Space `DineshAI/hCIBCAS1Hi`, followed by a hash-verified download,
then a fast-forward mirror of the published text paths to GitHub `main`.

## Experiment tree and winning branch

The stacked lineage is baseline theory → Claim 4 official wall-clock suite →
Claim 5 2D benchmark → Claim 6 paper-step route → executable morphology route
→ one-class PCA falsification route → evaluator-visible presentation child.
The winning formal experiment is `f2055881-2a9c-477a-a02d-907a67d7288f`,
branch `orx/claim-6-one-class-digit-manifold-falsification-r`, Git `990cb3d8afd53accb03a9e48f0c57e2842137785`,
run `199652d8-ec32-4192-a79f-d76f5ea9a46f`. The release child is presentation-only.

## Commands and compute

Every formal node inherited exactly:

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

Release checks:

```bash
UV_THREADPOOL_SIZE=1 uv run --frozen python release/verify_release.py
marimo check --strict notebooks/sosmc_reproduction.py
UV_THREADPOOL_SIZE=1 uv run --frozen python scripts/build_release.py
UV_THREADPOOL_SIZE=1 uv run --frozen python scripts/audit_candidate.py release/space
```

Formal runtime was 7,644.5407 seconds on Hugging Face `cpu-upgrade`, estimated
before launch as >1 core with uncertain runtime and allocated eight CPU cores.
CUDA was absent. Hugging Face cost is not exposed by the run log and is
therefore not guessed. Short packaging checks ran locally on one core.

## Evaluator-visible visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | pages/claim-1/page.md | yes | yes | yes | yes | yes | yes | VERIFIED |
| 2 | pages/claim-2/page.md | yes | yes | yes | yes | yes | yes | VERIFIED |
| 3 | pages/claim-3/page.md | yes | yes | yes | yes | yes | yes | VERIFIED |
| 4 | pages/claim-4/page.md | yes | yes | yes | yes | yes | yes | VERIFIED |
| 5 | pages/claim-5/page.md | yes | yes | yes | yes | yes | yes | VERIFIED |
| 6 | pages/claim-6/page.md | yes | yes | yes | yes | yes | yes | FALSIFIED |

## Evidence paths

- `release/space/pages/current-verification/page.md`
- `release/space/pages/claim-1/page.md` through `claim-6/page.md`
- `release/space/evidence/claim_*/`
- `release/space/raw/full_results.json`
- `release/space/verify_release.py`
- `.openresearch/artifacts/claim_*/`
- `reports/sosmc/report.md`

The old/new subset proof is `release/HISTORICAL_SUBSET_CHECK.txt`. The exact
text upload list is `release/HF_UPLOAD_ALLOWLIST.txt`; its hashes are in
`release/HF_UPLOAD_MANIFEST.sha256`.
