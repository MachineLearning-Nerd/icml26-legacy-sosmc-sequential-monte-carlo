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
