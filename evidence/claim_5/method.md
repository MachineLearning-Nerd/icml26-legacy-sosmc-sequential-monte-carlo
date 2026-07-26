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

At each fresh-evaluation iteration, raw evidence records reward, quadrature KL,
the paper's objective `reward - beta_KL * KL`, and the particle estimate.
An independent checker compares best objectives by paired dataset/seed,
computes particle-to-fresh RMSE, evaluates the large-beta control, and swaps
method labels as a negative control.

Fresh evaluation uses 500 samples, 2,000 ULA steps, and 200 burn-in steps.
Runtime calibration on the required CPU backend showed that even the authors'
documented reduced evaluator (1,000 samples, 5,000 steps, 500 burn-in) took
about 36 minutes to reach the first step-0 progress record, making the
six-method suite impossible within the four-hour job cap. Three bounded
evaluations are taken over 1,001 outer iterations, so tracking is observed at
start, midpoint, and endpoint (steps 0, 500, and 1000). This evaluator
substitution is treated as a material uncertainty and must be sensitivity
checked before a final Claim 5 acceptance.
