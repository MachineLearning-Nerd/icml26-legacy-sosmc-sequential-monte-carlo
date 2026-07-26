# Claim 5 limitations and deviations

- The experiment uses the authors' supplied pretrained EBM checkpoints rather
  than repeating the 200-epoch PCD pretraining.
- The saved checkpoint configs specify CUDA. They are loaded with the
  notebook's documented `device="cpu"` override to satisfy the campaign's
  CPU-only compute contract.
- The lower-half-plane reward is tested on all three datasets; the upper,
  left, and right panels are not part of this first benchmark node.
- This runtime-calibrated shard uses seed 0 at `beta_KL=0.25`. The parent
  three-seed design was cancelled after runtime calibration projected 6–7
  hours against a fixed 4-hour job cap, before any method endpoint was
  observed. A subsequent eight-method design was also cancelled before an
  endpoint when a second calibration projected more than four hours.
- The `beta_KL=5` circles control is split into a child node so the six
  small-beta methods fit within one job's four-hour cap.
- The paper's illustrative fresh evaluation uses 20,000 ULA steps and 15,000
  burn-in steps. This run uses the authors' own reduced-evaluation setting of
  5,000 and 500, respectively.
- Fresh evaluation is recorded at steps 0, 500, and 1000. An attempted
  five-point schedule was cancelled before its first endpoint after runtime
  calibration showed the six-method suite would exceed the four-hour cap.
- A passing result is direct evidence for the scoped contract, not a claim
  that every panel of the paper's full dataset/reward/beta grid was rerun.
- The cumulative runner intentionally does not accept Claim 5 from this
  single-seed shard.
