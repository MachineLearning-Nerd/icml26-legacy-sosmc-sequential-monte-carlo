# Efficient Stochastic Optimisation via Sequential Monte Carlo — reproduction

## Collection classification and audit boundary

This repository is a **legacy/source workspace** for *Efficient Stochastic Optimisation via Sequential Monte Carlo*
(arXiv `2601.22003`, OpenReview `hCIBCAS1Hi`). It is preserved
separately from the standardized canonical record at
[`icml26-sosmc-sequential-monte-carlo`](https://github.com/MachineLearning-Nerd/icml26-sosmc-sequential-monte-carlo).

The claim results and scores recorded below are historical results of this
workspace. They are not new paper-level verifications performed while
organizing the collection. The collection audit did not run the scientific
implementation; the canonical record documents its own scoped status and
limitations.

### How the historical claim evidence is produced

The claim table and experiment log below are the authoritative mapping from
each paper claim to its producer, command, control, and evidence artifact. In
this workspace, the SOSMC claim runners and independent checkers emit claim-specific raw outputs, controls, and `EVAL.md` artifacts under the committed report/logbook surfaces described in the experiment table.

The former `orx/*` branches are historical workstreams, not additional final
publication claims. Their purposes and tips are preserved in
[`BRANCH_AUDIT.md`](BRANCH_AUDIT.md). Citation and author acknowledgment
details are in [`CITATION.cff`](CITATION.cff) and
[`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo/blob/main/notebooks/sosmc_reproduction.py)

This campaign tests all six judged claims from
[arXiv:2601.22003](https://arxiv.org/abs/2601.22003). The cumulative CPU run
supports Claims 1–5 and finds a controlled counterexample to Claim 6 in the
authors' executable MNIST sweep. These are reproduction verdicts and a score
forecast—not a new judge score. The live score remains **3/12** until the judge
evaluates the updated Hugging Face Space.

The strongest empirical comparisons are:

| Claim | Paper result | Observed result | Assessment |
|---|---|---|---|
| 4, Gaussian-mixture reward tuning | SOSMC beats ImpDiff in four Table 2 settings; lower variability than SOUL in the specified case | SOSMC mean exceeds ImpDiff in all four settings over 10 seeds; smallest difference `+0.0000566`; specified SOUL controls pass | VERIFIED |
| 5, 2D EBM | Higher objective at small beta and comparable at large beta; particle reward tracks fresh reward | Circles best-objective difference `+0.004927` at beta `0.25`; `+0.0000319` at beta `5`; tracking RMSE `0.005658` vs `0.200299` | VERIFIED on the direct illustrative case |
| 6, MNIST | Fresh reward rises without reward hacking across three rewards and four beta values | Reward rises in all 12 cells, but three cells' digit-manifold median intervals exceed a fixed `4.170981` limit | FALSIFIED for the finite executable sweep |

The full run used Hugging Face `cpu-upgrade`, eight allocated CPU cores, no
GPU, and 7,644.54 seconds. The one-dataset/one-seed direct scope of Claim 5 and
the quantitative interpretation of “digit-like” in Claim 6 are explicit
limitations.

- [Illustrated technical report](reports/sosmc/report.md)
- [Tutorial-style marimo notebook](notebooks/sosmc_reproduction.py)
- [Evaluator-visible release report](release/RELEASE_REPORT.md)
- [Current Space evidence bundle](release/space/pages/current-verification/page.md)

## Experiment log

Every formal node used the same locked command:
`uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Publication surface | Not run as an experiment (publication surface) | README, report, notebook, and exact published text mirror | Local packaging only |
| [`orx/judged-baseline-plus-exact-theorem-calibration`](https://github.com/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo/tree/orx/judged-baseline-plus-exact-theorem-calibration) | Frozen proof-level baseline for Claims 1–3 | `uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run` | Claims 1–3 verified; 53 s | Local CPU, one core |
| [`orx/claim-4-official-wall-clock-langevin-suite`](https://github.com/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo/tree/orx/claim-4-official-wall-clock-langevin-suite) | Official four-setting wall-clock suite | `uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run` | Claim 4 verified; 8m01s | Hugging Face `cpu-upgrade` |
| [`orx/claim-5-circles-large-beta-comparability`](https://github.com/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo/tree/orx/claim-5-circles-large-beta-comparability) | Direct circles small/large-beta and tracking check | `uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run` | Claim 5 verified; 35m53s | Hugging Face `cpu-upgrade` |
| [`orx/claim-6-mnist-cumulative-paper-step-route`](https://github.com/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo/tree/orx/claim-6-mnist-cumulative-paper-step-route) | Written-paper step size and learned feature audit | `uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run` | BLOCKED: learned feature criterion rejected its controls | Hugging Face `cpu-upgrade` |
| [`orx/claim-6-executable-sweep-morphology-route`](https://github.com/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo/tree/orx/claim-6-executable-sweep-morphology-route) | Authors' executed configuration plus morphology | `uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run` | BLOCKED: morphology was not a complete perceptual certificate | Hugging Face `cpu-upgrade` |
| [`orx/claim-6-one-class-digit-manifold-falsification-r`](https://github.com/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo/tree/orx/claim-6-one-class-digit-manifold-falsification-r) | Independently calibrated one-class PCA route and cumulative regression | `uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run` | Claims 1–5 VERIFIED; Claim 6 FALSIFIED; 2h08m | Hugging Face `cpu-upgrade`, 8 CPUs |

## Reproduce

Python 3.12, dependencies, and hashes are pinned by `pyproject.toml` and
`uv.lock`.

```bash
uv sync --frozen --no-dev
uv run --frozen python -m sosmc_repro.run
```

The full run is long and CPU-heavy. Use the existing evidence for inspection;
do not rerun it on a single-core interactive machine. The marimo article is
pre-populated with the result:

```bash
uv run marimo edit notebooks/sosmc_reproduction.py
uv run marimo run notebooks/sosmc_reproduction.py
```
