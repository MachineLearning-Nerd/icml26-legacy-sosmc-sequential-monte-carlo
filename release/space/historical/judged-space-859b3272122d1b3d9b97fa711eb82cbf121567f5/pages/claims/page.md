# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_8dad23e7c0de", "created_at": "2026-07-26T13:50:45+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. The Sequential Optimization via SMC (SOSMC) framework replaces expensive inner MCMC sampling loops with a sequential Monte Carlo particle population that is reused and reweighted across optimization iterations, as formalized in Algorithm 1 (Section 3.1, Algorithm 1).
2. Proposition 2 establishes a linear convergence rate for the idealized SOSMC iteration under mu-Polyak-Lojasiewicz and L-smooth loss assumptions, showing l(theta_k) - inf(l) <= (1-gamma*mu)^k * (l(theta_0) - inf(l)) for step size gamma <= 1/L (Proposition 2).
3. Propositions 3 and 4 show the effective sample size of the SMC weights decays exponentially in the squared gradient norm and step size, ESS_infinity(gamma) = N*exp(-gamma^2*||grad l||^2) for Gaussian targets, motivating an adaptive step-size/resampling scheme (Section 4.2, Propositions 3-4).
4. On Langevin reward-tuning of energy-based models with non-differentiable reward functions, SOSMC-ULA outperforms the ImpDiff baseline and achieves reduced variance compared to single-chain SOUL (Section 5.1).
5. On 2D EBM benchmark datasets, SOSMC achieves higher objective values than ImpDiff for small regularization strengths, with particle-based reward estimates that track true expectations throughout optimization (Section 5.2).
6. On MNIST, SOSMC remains robust in high dimensions even under mismatch between the pretraining and tuning kernels, without exhibiting reward hacking (Section 5.3).
