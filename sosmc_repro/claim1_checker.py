from __future__ import annotations

import copy
import math
from typing import Any


GRADIENT_RELATIVE_L2_TOLERANCE = 1e-6
GRADIENT_MAX_ABSOLUTE_TOLERANCE = 1e-6
WEIGHT_SUM_TOLERANCE = 1e-6
NONUNIFORM_WEIGHT_STD_MINIMUM = 1e-12


def _check(trace: dict[str, Any]) -> dict[str, Any]:
    iterations = trace.get("iterations", [])
    gradient_checks = trace.get("gradient_checks", [])
    continuity = [
        iterations[index]["pre_particle_sha256"]
        == iterations[index - 1]["post_particle_sha256"]
        for index in range(1, len(iterations))
    ]
    one_proposal_per_iteration = [
        row.get("proposal_calls", 0) == 1 for row in iterations
    ]
    candidate_weights = [
        row["candidate_weights"]
        for row in iterations
        if "candidate_weights" in row
    ]
    normalized_weights = [
        abs(row["sum"] - 1.0) <= WEIGHT_SUM_TOLERANCE
        and 1.0 <= row["ess"] <= trace["n_particles"] + 1e-6
        for row in candidate_weights
    ]
    nonuniform_iterations = sum(
        row["std"] > NONUNIFORM_WEIGHT_STD_MINIMUM
        and row["max"] > row["min"]
        for row in candidate_weights
    )
    gradients_match = [
        math.isfinite(row["relative_l2_error"])
        and math.isfinite(row["max_absolute_error"])
        and row["relative_l2_error"]
        <= GRADIENT_RELATIVE_L2_TOLERANCE
        and row["max_absolute_error"]
        <= GRADIENT_MAX_ABSOLUTE_TOLERANCE
        for row in gradient_checks
    ]
    reference = trace.get("reference_initialization", {})
    reference_reused_only_for_equal_pair = (
        reference.get("cache_misses") == 1
        and reference.get("cache_hits") == 1
        and reference.get("reference_parameters_bitwise_equal") is True
        and reference.get("sampler_configuration_equal") is True
    )
    checks = {
        "at_least_three_consecutive_outer_iterations": len(iterations) >= 3,
        "particle_population_carried_between_iterations": (
            len(continuity) >= 2 and all(continuity)
        ),
        "exactly_one_ula_proposal_per_outer_iteration": (
            len(one_proposal_per_iteration) >= 3
            and all(one_proposal_per_iteration)
        ),
        "candidate_weights_normalized_and_valid": (
            len(normalized_weights) >= 3 and all(normalized_weights)
        ),
        "nonuniform_candidate_weights_in_two_iterations": (
            nonuniform_iterations >= 2
        ),
        "independent_weighted_gradient_recomputation": (
            len(gradients_match) >= 2 and all(gradients_match)
        ),
        "paired_reference_initialization_guarded": (
            reference_reused_only_for_equal_pair
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "continuity": continuity,
        "nonuniform_iteration_count": nonuniform_iterations,
        "gradient_tolerances": {
            "relative_l2": GRADIENT_RELATIVE_L2_TOLERANCE,
            "max_absolute": GRADIENT_MAX_ABSOLUTE_TOLERANCE,
        },
    }


def evaluate(trace: dict[str, Any]) -> dict[str, Any]:
    direct = _check(trace)
    corrupted = copy.deepcopy(trace)
    if len(corrupted.get("iterations", [])) >= 2:
        corrupted["iterations"][1][
            "pre_particle_sha256"
        ] = "fresh-population-control"
    negative = _check(corrupted)
    negative_control = {
        "name": "replace the carried population with a fresh population",
        "expected": "the Algorithm 1 continuity contract fails",
        "observed_passed": negative["passed"],
        "failed_for_intended_reason": (
            not negative["checks"][
                "particle_population_carried_between_iterations"
            ]
        ),
    }
    passed = (
        direct["passed"]
        and not negative_control["observed_passed"]
        and negative_control["failed_for_intended_reason"]
    )
    return {
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "passed": passed,
        "direct_checks": direct,
        "negative_control": negative_control,
    }
