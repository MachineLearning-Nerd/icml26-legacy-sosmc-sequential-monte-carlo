# Claim 5 limitations and deviations

- The experiment uses the authors' supplied pretrained EBM checkpoints rather
  than repeating the 200-epoch PCD pretraining.
- The saved checkpoint configs specify CUDA. They are loaded with the
  notebook's documented `device="cpu"` override to satisfy the campaign's
  CPU-only compute contract.
- The self-contained verifier uses the paper's lower-half-plane reward on
  circles at seed 0 and `beta_KL` values 0.25 and 5. The separate circles,
  two-moons, and blobs small-beta shards remain visible as robustness checks.
  Upper, left, and right reward panels were not rerun.
- The paper evaluates fresh reward using long ULA chains. Because the state
  space is two dimensional, this verifier instead computes normalized
  dense-grid quadrature for both reward and KL. It audits resolution and
  truncation-domain sensitivity, but does not reproduce chain mixing time.
- A passing result verifies the paper's stated finite illustrative behavior,
  whose wording says higher small-beta contours occur "in cases." It does not
  establish a universal advantage over every dataset, reward orientation,
  beta value, or seed.
- Stochastic replication beyond seed 0 remains a limitation. The methods use
  guarded identical reference particles and shared noise to make the paired
  finite comparison exact for that seed.
