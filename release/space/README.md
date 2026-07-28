---
title: "SOSMC claim-by-claim reproduction"
emoji: 🎯
colorFrom: green
colorTo: red
sdk: static
pinned: false
tags:
 - trackio-logbook
 - icml2026-repro
 - paper-hCIBCAS1Hi
---

# SOSMC claim-by-claim reproduction

**Current evaluator entrypoint.** Claims 1–5 are `VERIFIED`; Claim 6 is
`FALSIFIED` under the exact executable MNIST sweep. These are reproduction
verdicts and a score forecast—not a live judge result. The live score remains
**3/12** until the judge evaluates this revision.

Start with the [current verification](pages/current-verification/page.md), then
open the [six claim pages](pages/index.md), the
[visual report](report.md), or run the dependency-free
[`verify_release.py`](verify_release.py).

Fixed formal command:

```bash
uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run
```

The immutable prior judged revision `859b3272122d1b3d9b97fa711eb82cbf121567f5` is preserved under
[`historical/judged-space-859b3272122d1b3d9b97fa711eb82cbf121567f5/`](historical/judged-space-859b3272122d1b3d9b97fa711eb82cbf121567f5/README.md).
Its old default verifier is labeled **Historical rejected baseline** in the
current navigation.
