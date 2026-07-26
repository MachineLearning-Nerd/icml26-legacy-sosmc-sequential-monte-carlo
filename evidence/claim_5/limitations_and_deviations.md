# Claim 5 limitations and deviations

- The experiment uses the authors' supplied pretrained EBM checkpoints rather
  than repeating the 200-epoch PCD pretraining.
- The saved checkpoint configs specify CUDA. They are loaded with the
  notebook's documented `device="cpu"` override to satisfy the campaign's
  CPU-only compute contract.
- The lower-half-plane reward is tested on all three datasets; the upper,
  left, and right panels are not part of this first benchmark node.
- Three seeds are used at `beta_KL=0.25`; the large-beta control at
  `beta_KL=5` uses circles and seed 0.
- The paper's illustrative fresh evaluation uses 20,000 ULA steps and 15,000
  burn-in steps. This run uses the authors' own reduced-evaluation setting of
  5,000 and 500, respectively.
- A passing result is direct evidence for the scoped contract, not a claim
  that every panel of the paper's full dataset/reward/beta grid was rerun.
