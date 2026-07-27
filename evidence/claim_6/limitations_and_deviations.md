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

Appendix E.3 writes the lower-half reward with a factor `1/2`; the notebook
helper omits that factor. This reproduction follows the paper's formula and
records the discrepancy. The optimizer, model, sampler, particle count,
iteration count, and regularization grid remain the authors' implementation.
