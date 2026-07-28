# Claim 5 method

The verifier executes definition cells 1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
and 21 from the authors' vendored `reward_tuning/ebms_2D/experiments.ipynb`
at upstream commit `62e4f8f07ae2705073388f5d2c4babf5c87b00be`. It calls the
notebook's own `run_experimental_trial` function and loads the supplied
`checkpoint_latest.pt` for each dataset.

The saved upstream `config.json` files record the authors' original CUDA
device. The verifier passes `device="cpu"` through the notebook's documented
`load_trainer` override; model architecture, weights, optimiser state, data,
and all scientific hyperparameters are unchanged.

At steps 0, 500, and 1000, raw evidence records the normalized dense-grid
quadrature reward and KL, the paper's objective `reward - beta_KL * KL`, and
the particle estimate. This deterministic 2D evaluator replaces the
notebook's long-chain approximation and removes fresh-sampling error. The
primary 400-by-400 grid on `[-6,6]^2` is independently checked with a
600-by-600 grid and a wider `[-8,8]^2` domain.

The self-contained circles node runs both `beta_KL=0.25` and `beta_KL=5`.
An independent checker tests the small-beta best-objective direction,
particle-to-grid reward RMSE, large-beta objective comparability, all grid
variants, and a reversed-label negative control. Separate two-moons and blobs
nodes are robustness shards rather than assumptions of the finite verifier.
