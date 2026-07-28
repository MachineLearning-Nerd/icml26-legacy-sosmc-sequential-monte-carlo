# Limitations and deviations

The paper's “no reward hacking” conclusion is qualitative and image-based.
Route 1's learned recognizer-feature support criterion was rejected because it
failed all three obvious non-digit controls. Route 2 uses simple,
classifier-independent pixel morphology. This is an independent quantitative
operationalization, not a claim that standard deviation, total variation, and
pooled-pixel support fully capture human perception. Sample figures will remain
visible in the final report, but they will not substitute for machine checks.

The formal terminal evaluator disables the notebook's repeated 50-step fresh
evaluations during tuning. It retains the exact 1,000 optimization iterations
and uses a stronger terminal evaluation: three paired 64-sample, 512-step runs
under the original pretraining sampler. This avoids using intermediate
evaluation randomness to change optimization and directly tests the paper's
terminal robustness statement.

Appendix E.3 writes `R_dark=-mean(x)` and the lower-half reward with a factor
`1/2`. The executed darkness sweep introduces a factor `1/2`, while the
notebook half-plane helper omits it. This second route follows the executable
notebook scalings. Route 1 followed the written formulas and remains preserved.
The optimizer, model, sampler, particle count, iteration count, and
regularization grid remain the authors' implementation.

Appendix E.3 reports an initial tuning step size of `5e-3`, whereas the released
notebook's beta sweep and saved outputs use `3e-3`. Route 1 followed the
paper-text value (`5e-3`). This second route follows the authors' executed value
(`3e-3`). Neither recorded route is changed retrospectively after its run.
