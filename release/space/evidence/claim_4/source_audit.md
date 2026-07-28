# Claim 4 source audit

- Source: <https://ar5iv.labs.arxiv.org/html/2601.22003#S5.SS1>
- Retrieval date: 2026-07-26
- Paper HTML SHA-256:
  `ec13f8ac302898767ba1aedeefd94742c126415bbdc3db27b23a8ae89e6c7128`
- Exact empirical scope: Figure 1 and Appendix E.1/Table 2, 10 runs, Adam and
  SGD, fixed device-synchronized wall-clock budget, equal kernel applications
  per outer iteration, persistent particles, and non-differentiable rewards.

The paper's actual wording is narrower than “SOSMC always outperforms”: it reports
swift convergence relative to ImpDiff, greater SOUL variability in Figure 1a,
a SOUL mode-transition failure in Figure 1b, and robust Metropolis-corrected
variants. This contract tests the SOSMC-ULA/ImpDiff/SOUL subset across all four
Table 2 Adam settings.
