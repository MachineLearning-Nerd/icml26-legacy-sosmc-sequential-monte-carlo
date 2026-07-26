# Limitations and deviations

- CPU-only execution changes the number of outer steps completed in two seconds
  relative to the paper's personal computer/Colab hardware.
- Time-budget mode is inherently not bitwise deterministic even with fixed
  seeds; the official notebook explicitly documents this. A sibling experiment
  performs a deterministic step-matched calibration.
- This branch tests Adam because Table 2 reports both optimizers and the headline
  Figure 1b uses Adam; the cumulative campaign retains the precise scope.

