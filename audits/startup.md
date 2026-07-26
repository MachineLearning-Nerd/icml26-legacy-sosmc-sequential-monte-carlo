# Startup audit

Audit date: 2026-07-26 (Asia/Kolkata).

- Repository `main` and `origin/main`: `ea2f7db8148303e8512b3b6ad61534769b8dbaed`
- OpenResearch project: `b16b3d5b-a97e-45eb-b87d-f01af10c5c5d`
- Initial experiment tree and runs: empty
- Initial working tree: detached and clean
- Free disk at startup: 26 GiB
- Paper source: <https://ar5iv.labs.arxiv.org/html/2601.22003>
- Retrieval User-Agent:
  `OpenResearch-Reproduction/1.0 (+https://github.com/MachineLearning-Nerd/icml26-repro-hCIBCAS1Hi-efficient-stochastic-optimisation-via-sequential-monte-carlo)`
- Paper HTML SHA-256:
  `ec13f8ac302898767ba1aedeefd94742c126415bbdc3db27b23a8ae89e6c7128`
- Official code: <https://github.com/akyildiz-group/SOSMC>
- Inspected upstream commit: `62e4f8f07ae2705073388f5d2c4babf5c87b00be`
- Live verdict dataset: `ICML-2026-agent-repro/verdicts`
- Verdict dataset revision: `950caa071d937ed1422c8c93a2aa1f8ffb159331`
- Verdict JSON SHA-256:
  `fbbc4a30e15512832eff77ef58cafcec3a001597682759dd146ee2aebe5b3e09`
- Filter used: `space_id == "DineshAI/hCIBCAS1Hi"`; exactly one row matched
- Judged Space revision:
  `DineshAI/hCIBCAS1Hi@859b3272122d1b3d9b97fa711eb82cbf121567f5`
- Protected judged file count: 17
- Fixed run command:
  `uv sync --frozen --no-dev && uv run --frozen python -m sosmc_repro.run`
- Compute policy: local only for certain one-core tasks below five minutes;
  Hugging Face `cpu-upgrade` for multicore, longer, or uncertain CPU tasks.

Environment variable names were audited without printing values. Names indicated
that Hugging Face and GitHub authentication are available through the configured
OpenResearch integration; no secret material was copied to evidence.

