# Limitations and deviations

The paper's “no reward hacking” conclusion is qualitative and image-based. The
recognizer/support-distance criterion is an independent quantitative
operationalization, not a claim that any classifier metric fully captures
human perception. Sample figures will remain visible in the final report, but
they will not substitute for the machine checks.

The formal terminal evaluator disables the notebook's repeated 50-step fresh
evaluations during tuning. It retains the exact 1,000 optimization iterations
and uses a stronger terminal evaluation: three paired 64-sample, 512-step runs
under the original pretraining sampler. This avoids using intermediate
evaluation randomness to change optimization and directly tests the paper's
terminal robustness statement.

Appendix E.3 writes `R_dark=-mean(x)` and the lower-half reward with a factor
`1/2`. The executed darkness sweep introduces a factor `1/2`, while the
notebook half-plane helper omits it. This reproduction follows the paper's
formulas and records both discrepancies. The optimizer, model, sampler,
particle count, iteration count, and regularization grid remain the authors'
implementation.

Appendix E.3 reports an initial tuning step size of `5e-3`, whereas the released
notebook's beta sweep and saved outputs use `3e-3`. This primary cumulative
node follows the paper-text value (`5e-3`). If the resulting evidence remains
interpretation sensitive, the already committed executable-code route at
`3e-3` is retained as a separate child; neither value is changed
retrospectively after a run.
