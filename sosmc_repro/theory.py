from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
import sympy as sp


def verify_proposition_1() -> dict[str, Any]:
    """Machine-check the exact one-step implication behind Proposition 1.

    The proof certificate is algebraic. It uses only the descent lemma,
    gamma <= 1/L, and the paper's PL convention
    ||grad l||^2 >= 2 mu (l - inf l).
    """

    gamma, L, mu, gap, grad_sq = sp.symbols(
        "gamma L mu gap grad_sq", positive=True
    )
    descent_rhs = -gamma * (1 - L * gamma / 2) * grad_sq
    boundary_rhs = -gamma * grad_sq / 2
    pl_rhs = -gamma * mu * gap

    boundary_identity = sp.simplify(
        descent_rhs - boundary_rhs + gamma * (1 - L * gamma) * grad_sq / 2
    )
    pl_identity = sp.simplify(boundary_rhs - pl_rhs + gamma * (grad_sq - 2 * mu * gap) / 2)

    # Exact rational witness over a complete grid of quadratic eigenvalues.
    violations: list[dict[str, str]] = []
    for l_num in range(1, 13):
        for mu_num in range(1, l_num + 1):
            Lq = Fraction(l_num, 1)
            muq = Fraction(mu_num, 1)
            for gamma_fraction in (Fraction(1, 4), Fraction(1, 2), Fraction(1, 1)):
                gq = gamma_fraction / Lq
                for eigen_num in range(mu_num, l_num + 1):
                    lam = Fraction(eigen_num, 1)
                    contraction = (1 - gq * lam) ** 2
                    bound = 1 - gq * muq
                    if contraction > bound:
                        violations.append(
                            {
                                "L": str(Lq),
                                "mu": str(muq),
                                "gamma": str(gq),
                                "eigenvalue": str(lam),
                            }
                        )

    # Negative controls: remove each indispensable assumption.
    gamma_bad = Fraction(5, 2) / Fraction(10, 1)
    quadratic_bad_contraction = (1 - gamma_bad * 10) ** 2
    quadratic_claimed_bound = 1 - gamma_bad * 1
    non_pl_ratios = [
        float((4 * x**6) / (2 * x**4)) for x in (1.0, 0.1, 0.01, 0.001)
    ]

    controls = {
        "gamma_above_1_over_L_breaks_certificate": bool(
            quadratic_bad_contraction > quadratic_claimed_bound
        ),
        "gamma_bad": str(gamma_bad),
        "bad_contraction": str(quadratic_bad_contraction),
        "bad_claimed_bound": str(quadratic_claimed_bound),
        "non_pl_gradient_ratio_tends_to_zero": bool(
            all(a > b for a, b in zip(non_pl_ratios, non_pl_ratios[1:]))
            and non_pl_ratios[-1] < 1e-4
        ),
        "non_pl_ratios": non_pl_ratios,
    }

    passed = (
        boundary_identity == 0
        and pl_identity == 0
        and not violations
        and all(
            controls[key]
            for key in (
                "gamma_above_1_over_L_breaks_certificate",
                "non_pl_gradient_ratio_tends_to_zero",
            )
        )
    )
    return {
        "claim": "Paper Proposition 1 linear convergence under A1",
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "symbolic_certificate": {
            "descent_to_half_step_identity": str(boundary_identity),
            "pl_substitution_identity": str(pl_identity),
            "assumptions": [
                "L-smooth loss",
                "mu-PL: ||grad l||^2 >= 2 mu (l-inf l)",
                "0 < gamma <= 1/L",
                "idealized SOSMC update equals exact gradient descent by Lemma 1",
            ],
        },
        "exact_quadratic_grid": {
            "L_values": 12,
            "mu_per_L": "all integers 1..L",
            "gamma_fractions_of_1_over_L": ["1/4", "1/2", "1"],
            "eigenvalues": "all integers mu..L",
            "violations": violations,
        },
        "negative_controls": controls,
        "passed": passed,
    }


def _empirical_chi2(
    delta: np.ndarray, covariance: np.ndarray, seed: int, n: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(np.zeros(delta.size), covariance, size=n)
    precision = np.linalg.inv(covariance)
    log_ratio = samples @ precision @ delta - 0.5 * delta @ precision @ delta
    # chi2(p||q) + 1 = E_q[(p/q)^2].
    empirical = float(np.mean(np.exp(2.0 * log_ratio)))
    exact = float(np.exp(delta @ precision @ delta))
    return empirical, exact


def verify_equation_19(seed: int) -> dict[str, Any]:
    """Verify the exact Gaussian identity and its domain, independently."""

    t, d, sigma = sp.symbols("t d sigma", real=True, positive=True)
    q = sp.exp(-t**2 / (2 * sigma**2)) / sp.sqrt(2 * sp.pi * sigma**2)
    ratio = sp.exp((2 * t * d - d**2) / (2 * sigma**2))
    integral = sp.integrate(sp.simplify(q * ratio**2), (t, -sp.oo, sp.oo))
    symbolic_identity = sp.simplify(integral - sp.exp(d**2 / sigma**2))

    rng = np.random.default_rng(seed)
    dimensions = (1, 2, 5, 11)
    rows = []
    for dim in dimensions:
        a = rng.normal(size=(dim, dim))
        covariance = a @ a.T + 2.0 * np.eye(dim)
        direction = rng.normal(size=dim)
        precision = np.linalg.inv(covariance)
        norm = math.sqrt(float(direction @ precision @ direction))
        delta = direction * (0.45 / max(norm, 1e-12))
        empirical, exact = _empirical_chi2(delta, covariance, seed + dim, 250_000)
        rows.append(
            {
                "dimension": dim,
                "empirical_chi2_plus_one": empirical,
                "exact_chi2_plus_one": exact,
                "relative_error": abs(empirical - exact) / exact,
            }
        )

    # Negative control: Equation 19 assumes equal covariance. For zero mean but
    # unequal covariance it would predict rho=N, although chi2 is nonzero.
    variance_p = 1.2
    variance_q = 1.0
    chi2_plus_one_mismatch = variance_q / math.sqrt(
        variance_p * (2 * variance_q - variance_p)
    )
    mismatch_chi2 = chi2_plus_one_mismatch - 1.0
    controls = {
        "unequal_covariance_invalidates_equal_covariance_formula": mismatch_chi2 > 0.01,
        "variance_p": variance_p,
        "variance_q": variance_q,
        "actual_chi2": mismatch_chi2,
        "equation_19_wrongly_extended_chi2": 0.0,
    }

    max_relative_error = max(row["relative_error"] for row in rows)
    passed = (
        symbolic_identity == 0
        and max_relative_error < 0.025
        and controls["unequal_covariance_invalidates_equal_covariance_formula"]
    )
    return {
        "claim": "Paper Equation 19 for equal-covariance Gaussian targets",
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "symbolic_univariate_whitened_identity": str(symbolic_identity),
        "independent_monte_carlo": rows,
        "max_relative_error": max_relative_error,
        "negative_controls": controls,
        "passed": passed,
    }

