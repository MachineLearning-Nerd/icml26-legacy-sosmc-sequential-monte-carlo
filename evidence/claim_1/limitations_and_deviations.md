# Limitations and deviations

The frozen root baseline has no faithful EBM execution. This child supersedes
that blocked result with a trace from the official checkpointed 2D EBM code.
Only the first three of 1,001 outer iterations are retained at tensor-hash
granularity to bound evidence size; the full tuning run still executes.

The paired ImpDiff/SOSMC reference draw is reused only after bitwise
reference-parameter and exact sampler-configuration checks. This avoids a second
identical 20,000-step initialization without changing the SOSMC particle
population or any outer update.
