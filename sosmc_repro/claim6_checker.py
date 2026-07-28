from __future__ import annotations

import copy
from typing import Any


REWARDS = ("bright", "dark", "lower_half")
BETAS = (5.0, 2.0, 1.0, 0.5)
CLASSIFIER_ACCURACY_MINIMUM = 0.97
DIGIT_DISTANCE_MULTIPLIER = 1.5
MORPHOLOGY_RETENTION_MINIMUM = 0.5


def _check(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("raw_rows", [])
    by_key = {
        (row["reward"], float(row["beta_kl"])): row for row in rows
    }
    expected = {(reward, beta) for reward in REWARDS for beta in BETAS}
    complete_grid = set(by_key) == expected
    reward_checks: dict[str, bool] = {}
    structure_checks: dict[str, bool] = {}
    structure_audit: dict[str, Any] = {}
    beta_order_audit: dict[str, Any] = {}
    recognizer = result.get("recognizer", {})
    real_morphology = recognizer.get("real_test_morphology", {})
    real_support_q95 = float(
        real_morphology
        .get("multiscale_pixel_support_distance", {})
        .get("q95", float("inf"))
    )
    real_pixel_std_median = float(
        real_morphology
        .get("pixel_standard_deviation", {})
        .get("median", 0.0)
    )
    real_total_variation_median = float(
        real_morphology
        .get("total_variation", {})
        .get("median", 0.0)
    )
    feature_real_q95 = float(
        result.get("recognizer", {})
        .get("real_test_support_distance", {})
        .get("q95", float("inf"))
    )
    for reward in REWARDS:
        ordered_means = []
        for beta in BETAS:
            key = f"{reward}:beta={beta:g}"
            row = by_key.get((reward, beta))
            if row is None:
                reward_checks[key] = False
                structure_checks[key] = False
                continue
            interval = row["paired_reward_difference"]
            reward_checks[key] = (
                interval["mean"] > 0.0
                and interval["ci95_low"] > 0.0
                and row["post_reward"]["mean"]
                > row["baseline_reward"]["mean"]
            )
            baseline = row["digit_structure"]["baseline"]
            tuned = row["digit_structure"]["tuned"]
            hacked = row["digit_structure"]["reward_maximizing_control"]
            baseline_morphology = baseline[
                "classifier_independent_morphology"
            ]
            tuned_morphology = tuned[
                "classifier_independent_morphology"
            ]
            hacked_morphology = hacked[
                "classifier_independent_morphology"
            ]
            support_limit = max(
                real_support_q95,
                DIGIT_DISTANCE_MULTIPLIER
                * baseline_morphology[
                    "multiscale_pixel_support_distance"
                ]["median"],
            )
            pixel_std_floor = MORPHOLOGY_RETENTION_MINIMUM * min(
                real_pixel_std_median,
                baseline_morphology[
                    "pixel_standard_deviation"
                ]["median"],
            )
            total_variation_floor = (
                MORPHOLOGY_RETENTION_MINIMUM
                * min(
                    real_total_variation_median,
                    baseline_morphology["total_variation"]["median"],
                )
            )
            tuned_digit_like = (
                tuned_morphology["pixel_standard_deviation"]["median"]
                >= pixel_std_floor
                and tuned_morphology["total_variation"]["median"]
                >= total_variation_floor
                and tuned_morphology[
                    "multiscale_pixel_support_distance"
                ]["median"]
                <= support_limit
            )
            hacked_digit_like = (
                hacked_morphology["pixel_standard_deviation"]["median"]
                >= pixel_std_floor
                and hacked_morphology["total_variation"]["median"]
                >= total_variation_floor
                and hacked_morphology[
                    "multiscale_pixel_support_distance"
                ]["median"]
                <= support_limit
            )
            hacked_reward_maximal = (
                hacked["reward_mean"]
                >= row["post_reward"]["maximum"] - 1e-7
            )
            structure_checks[key] = (
                tuned_digit_like
                and not hacked_digit_like
                and hacked_reward_maximal
            )
            structure_audit[key] = {
                "thresholds_calibrated_without_tuned_samples": {
                    "pixel_standard_deviation_median_minimum": (
                        pixel_std_floor
                    ),
                    "total_variation_median_minimum": (
                        total_variation_floor
                    ),
                    "multiscale_pixel_support_median_maximum": (
                        support_limit
                    ),
                },
                "tuned": tuned_morphology,
                "reward_maximizing_control": hacked_morphology,
                "tuned_digit_like": tuned_digit_like,
                "reward_maximizing_control_digit_like": (
                    hacked_digit_like
                ),
                "reward_maximizing_control_reward_maximal": (
                    hacked_reward_maximal
                ),
                "classifier_diagnostics_not_used_for_acceptance": {
                    "tuned_confidence_median": tuned[
                        "classifier_confidence_median"
                    ],
                    "tuned_predicted_class_count": tuned[
                        "predicted_class_count"
                    ],
                    "legacy_feature_support_median": tuned[
                        "support_distance_median"
                    ],
                    "legacy_real_feature_support_q95": (
                        feature_real_q95
                    ),
                },
            }
            ordered_means.append(row["post_reward"]["mean"])
        beta_order_audit[reward] = {
            "betas_high_to_low": list(BETAS),
            "post_reward_means": ordered_means,
            "nondecreasing_as_regularization_weakens": (
                len(ordered_means) == len(BETAS)
                and all(
                    right >= left
                    for left, right in zip(
                        ordered_means, ordered_means[1:]
                    )
                )
            ),
            "acceptance_note": (
                "Reported as a diagnostic; the paper's Claim 6 contract "
                "requires improvement in every finite cell, not strict "
                "monotonicity between noisy endpoints."
            ),
        }

    recognizer_valid = (
        recognizer.get("test_accuracy", 0.0)
        >= CLASSIFIER_ACCURACY_MINIMUM
    )
    kernel = result.get("kernel_mismatch_audit", {})
    mismatch_exact = (
        kernel.get("pretraining_sampler")
        == {
            "jitter_noise_std": 0.005,
            "gradient_clip": 0.03,
            "gradient_step_scale": 10.0,
            "state_clamp": [-1.0, 1.0],
            "transition_density": "not Gaussian",
        }
        and kernel.get("tuning_sampler")
        == {
            "kernel": "pure Gaussian ULA",
            "step_size": 0.003,
            "noise_scale": 1.0,
            "gradient_clip": None,
            "state_clamp": None,
            "jitter_noise": None,
            "steps_per_outer": 1,
        }
    )
    cache = result.get("initial_particle_cache", {})
    guarded_initialization = (
        cache.get("cache_misses") == 1
        and cache.get("cache_hits") == len(expected) - 1
        and cache.get("all_reference_parameters_bitwise_equal") is True
        and cache.get("all_sampler_configurations_equal") is True
    )
    checks = {
        "complete_three_reward_four_beta_grid": complete_grid,
        "fresh_reward_increase_ci_all_cells": (
            len(reward_checks) == len(expected)
            and all(reward_checks.values())
        ),
        "digit_structure_preserved_and_hack_controls_rejected": (
            len(structure_checks) == len(expected)
            and all(structure_checks.values())
        ),
        "independent_recognizer_valid": recognizer_valid,
        "pretraining_tuning_kernel_mismatch_exact": mismatch_exact,
        "shared_initialization_guarded": guarded_initialization,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reward_checks": reward_checks,
        "structure_checks": structure_checks,
        "structure_audit": structure_audit,
        "beta_order_audit": beta_order_audit,
        "fixed_thresholds": {
            "classifier_test_accuracy_minimum": (
                CLASSIFIER_ACCURACY_MINIMUM
            ),
            "digit_distance_multiplier": DIGIT_DISTANCE_MULTIPLIER,
            "morphology_retention_minimum": (
                MORPHOLOGY_RETENTION_MINIMUM
            ),
            "threshold_calibration": (
                "held-out real MNIST and pretrained EBM only; tuned "
                "samples and reward-maximizing controls are excluded"
            ),
        },
    }


def evaluate(result: dict[str, Any]) -> dict[str, Any]:
    direct = _check(result)
    swapped = copy.deepcopy(result)
    for row in swapped.get("raw_rows", []):
        difference = row["paired_reward_difference"]
        old_low = difference["ci95_low"]
        old_high = difference["ci95_high"]
        difference["mean"] = -difference["mean"]
        difference["ci95_low"] = -old_high
        difference["ci95_high"] = -old_low
        row["baseline_reward"], row["post_reward"] = (
            row["post_reward"],
            row["baseline_reward"],
        )
    swapped_check = _check(swapped)
    negative_control = {
        "name": "swap pretrained and tuned reward labels",
        "expected": "the every-cell fresh-reward improvement check fails",
        "observed_passed": swapped_check["passed"],
        "failed_for_intended_reason": (
            not swapped_check["checks"][
                "fresh_reward_increase_ci_all_cells"
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
