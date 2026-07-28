#!/usr/bin/env python3
"""Independent, dependency-free verifier for the published evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify(root: Path) -> None:
    result = json.loads((root / "raw" / "full_results.json").read_text())
    claims = result["claim_results"]
    expected = {
        "1": "VERIFIED",
        "2": "VERIFIED",
        "3": "VERIFIED",
        "4": "VERIFIED",
        "5": "VERIFIED",
        "6": "FALSIFIED",
    }
    require(result["summary"]["results"] == expected, "summary verdicts changed")
    require(result["summary"]["all_accepted_passed"], "cumulative suite failed")

    c1 = claims["1"]
    require(c1["passed"], "Claim 1 checker failed")
    for trial in c1["trials"]:
        checks = trial["independent_checker"]["direct_checks"]["checks"]
        require(all(checks.values()), "Claim 1 direct check failed")
        require(
            trial["independent_checker"]["negative_control"][
                "failed_for_intended_reason"
            ],
            "Claim 1 negative control did not fail",
        )

    c2 = claims["2"]
    require(c2["symbolic_certificate"]["descent_to_half_step_identity"] == "0",
            "Claim 2 descent identity is nonzero")
    require(c2["symbolic_certificate"]["pl_substitution_identity"] == "0",
            "Claim 2 PL identity is nonzero")
    require(not c2["exact_quadratic_grid"]["violations"],
            "Claim 2 exact grid has violations")
    require(
        c2["negative_controls"]["gamma_above_1_over_L_breaks_certificate"],
        "Claim 2 step-size negative control did not fail",
    )

    c3 = claims["3"]
    require(c3["symbolic_univariate_whitened_identity"] == "0",
            "Claim 3 symbolic identity is nonzero")
    require(c3["max_relative_error"] < 0.005,
            "Claim 3 Monte Carlo relative error is too large")
    require(
        c3["negative_controls"][
            "unequal_covariance_invalidates_equal_covariance_formula"
        ],
        "Claim 3 unequal-covariance control did not fail",
    )

    c4 = claims["4"]["independent_checker"]
    require(c4["passed"], "Claim 4 independent checker failed")
    require(
        all(c4["checks"]["sosmc_mean_exceeds_impdiff_all_four_settings"].values()),
        "Claim 4 direction failed",
    )
    require(c4["negative_control"]["failed_as_intended"],
            "Claim 4 reversed-label control did not fail")

    c5 = claims["5"]["independent_checker"]
    require(c5["passed"], "Claim 5 independent checker failed")
    require(all(c5["grid_sensitivity"]["positive_direction"].values()),
            "Claim 5 quadrature sensitivity failed")
    require(c5["tracking"]["SOSMC-ULA"]["rmse_particle_vs_fresh"]
            < c5["tracking"]["ImpDiff"]["rmse_particle_vs_fresh"],
            "Claim 5 tracking direction failed")
    require(c5["negative_control"]["failed_as_intended"],
            "Claim 5 reversed-label control did not fail")

    c6 = claims["6"]
    direct = c6["independent_checker"]["direct_checks"]
    require(c6["verdict"] == "FALSIFIED", "Claim 6 verdict changed")
    require(direct["falsification_passed"], "Claim 6 counterexample failed")
    require(all(direct["reward_checks"].values()),
            "Claim 6 reward improvement grid failed")
    require(
        set(direct["counterexample_cells"])
        == {"bright:beta=0.5", "lower_half:beta=1", "lower_half:beta=0.5"},
        "Claim 6 counterexample cells changed",
    )
    require(c6["negative_controls"]["label_swap"]["failed_for_intended_reason"],
            "Claim 6 label-swap control did not fail")
    print("PASS: Claims 1-5 VERIFIED; Claim 6 FALSIFIED; controls behaved as intended")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="release evidence root containing raw/full_results.json",
    )
    args = parser.parse_args()
    verify(args.root)


if __name__ == "__main__":
    main()
