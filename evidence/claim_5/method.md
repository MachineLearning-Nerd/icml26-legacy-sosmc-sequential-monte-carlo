# Claim 5 method

The verifier executes definition cells 1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
and 21 from the authors' vendored `reward_tuning/ebms_2D/experiments.ipynb`
at upstream commit `62e4f8f07ae2705073388f5d2c4babf5c87b00be`. It calls the
notebook's own `run_experimental_trial` function and loads the supplied
`checkpoint_latest.pt` for each dataset.

At each fresh-evaluation iteration, raw evidence records reward, quadrature KL,
the paper's objective `reward - beta_KL * KL`, and the particle estimate.
An independent checker compares best objectives by paired dataset/seed,
computes particle-to-fresh RMSE, evaluates the large-beta control, and swaps
method labels as a negative control.

Fresh evaluation uses the authors' documented reduced chain: 1,000 samples,
5,000 ULA steps, and 500 burn-in steps. Five evaluations are taken over 1,001
outer iterations so tracking is observed at steps 0, 250, 500, 750, and 1000.
