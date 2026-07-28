from __future__ import annotations

import copy
from typing import Any


REWARDS = ("bright", "dark", "lower_half")
BETAS = (5.0, 2.0, 1.0, 0.5)
CLASSIFIER_ACCURACY_MINIMUM = 0.97
DIGIT_DISTANCE_MULTIPLIER = 1.5
MORPHOLOGY_RETENTION_MINIMUM = 0.5


def _median_interval(values: list[float]) -> dict[str, float]:
    """Conservative normal-approximation order-statistic interval."""
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count < 16:
        return {
            "count": count,
            "median": float("nan"),
            "ci95_low": float("-inf"),
            "ci95_high": float("inf"),
        }
    half_width = 1.96 * (count * 0.25) ** 0.5
    low_index = max(0, int(count * 0.5 - half_width) - 1)
    high_index = min(
        count - 1, int(count * 0.5 + half_width) + 1
    )
    return {
        "count": count,
        "median": ordered[(count - 1) // 2],
        "ci95_low": ordered[low_index],
        "ci95_high": ordered[high_index],
        "method": (
            "95% binomial-normal order-statistic interval for the "
            "population median"
        ),
    }


def _check(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("raw_rows", [])
    by_key = {
        (row["reward"], float(row["beta_kl"])): row for row in rows
    }
    expected = {(reward, beta) for reward in REWARDS for beta in BETAS}
    complete_grid = set(by_key) == expected
    reward_checks: dict[str, bool] = {}
    structure_checks: dict[str, bool] = {}
    falsification_checks: dict[str, bool] = {}
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
    pca_real = recognizer.get(
        "pca_real_test_digit_manifold_score", {}
    )
    pca_shuffled = recognizer.get(
        "pca_shuffled_real_digit_manifold_score", {}
    )
    pca_real_q99 = float(pca_real.get("q99", float("inf")))
    pca_calibration_valid = (
        recognizer.get("pca_components") == 64
        and recognizer.get("pca_training_images") == 5000
        and recognizer.get("pca_seed") == 2026072722
        and 0.0 < pca_real_q99 < float("inf")
    )
    shuffled_control_checks: dict[str, bool] = {}
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
            baseline_pca = baseline[
                "pca_digit_manifold_score"
            ]
            tuned_pca = tuned["pca_digit_manifold_score"]
            hacked_pca = hacked["pca_digit_manifold_score"]
            pca_limit = max(
                pca_real_q99,
                float(baseline_pca.get("q95", float("inf"))),
            )
            tuned_pca_interval = _median_interval(
                tuned.get(
                    "pca_digit_manifold_score_values", []
                )
            )
            hacked_pca_interval = _median_interval(
                hacked.get(
                    "pca_digit_manifold_score_values", []
                )
            )
            pca_tuned_preserved = (
                float(tuned_pca.get("q95", float("inf")))
                <= pca_limit
            )
            pca_tuned_outside = (
                tuned_pca_interval["ci95_low"] > pca_limit
            )
            pca_hack_rejected = (
                hacked_pca_interval["ci95_low"] > pca_limit
            )
            shuffled_rejected = (
                float(pca_shuffled.get("q05", float("-inf")))
                > pca_limit
            )
            baseline_accepted = (
                float(baseline_pca.get("median", float("inf")))
                <= pca_limit
            )
            structure_checks[key] = (
                pca_tuned_preserved
                and pca_hack_rejected
                and shuffled_rejected
                and baseline_accepted
                and hacked_reward_maximal
            )
            falsification_checks[key] = (
                reward_checks[key]
                and pca_tuned_outside
                and pca_hack_rejected
                and shuffled_rejected
                and baseline_accepted
                and hacked_reward_maximal
            )
            shuffled_control_checks[key] = shuffled_rejected
            structure_audit[key] = {
                "one_class_pca_certificate": {
                    "threshold": pca_limit,
                    "threshold_rule": (
                        "max(held-out-real q99, pretrained-baseline "
                        "q95); tuned samples and controls excluded"
                    ),
                    "baseline": baseline_pca,
                    "tuned": tuned_pca,
                    "tuned_median_interval": tuned_pca_interval,
                    "reward_maximizing_control": hacked_pca,
                    "reward_maximizing_control_median_interval": (
                        hacked_pca_interval
                    ),
                    "pixel_shuffled_real_control": pca_shuffled,
                    "baseline_accepted": baseline_accepted,
                    "tuned_q95_within_digit_manifold": (
                        pca_tuned_preserved
                    ),
                    "tuned_median_outside_digit_manifold": (
                        pca_tuned_outside
                    ),
                    "reward_maximizer_rejected": (
                        pca_hack_rejected
                    ),
                    "unseen_pixel_shuffle_rejected": (
                        shuffled_rejected
                    ),
                },
                "route_2_morphology_diagnostic_not_used_for_route_3": {
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
        "one_class_pca_calibration_valid": pca_calibration_valid,
        "unseen_pixel_shuffle_rejected": (
            len(shuffled_control_checks) == len(expected)
            and all(shuffled_control_checks.values())
        ),
        "independent_recognizer_valid": recognizer_valid,
        "pretraining_tuning_kernel_mismatch_exact": mismatch_exact,
        "shared_initialization_guarded": guarded_initialization,
    }
    prerequisites_passed = all(checks.values())
    verification_passed = (
        prerequisites_passed
        and len(structure_checks) == len(expected)
        and all(structure_checks.values())
    )
    counterexample_cells = [
        key
        for key, established in falsification_checks.items()
        if established
    ]
    falsification_passed = (
        prerequisites_passed and bool(counterexample_cells)
    )
    passed = verification_passed or falsification_passed
    if falsification_passed:
        verdict = "FALSIFIED"
    elif verification_passed:
        verdict = "VERIFIED"
    else:
        verdict = "BLOCKED"
    return {
        "passed": passed,
        "verdict": verdict,
        "checks": checks,
        "reward_checks": reward_checks,
        "structure_checks": structure_checks,
        "falsification_checks": falsification_checks,
        "counterexample_cells": counterexample_cells,
        "verification_passed": verification_passed,
        "falsification_passed": falsification_passed,
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
            "pca_components": 64,
            "pca_training_images": 5000,
            "pca_seed": 2026072722,
            "pca_digit_manifold_score_threshold": (
                "max(held-out-real q99, pretrained-baseline q95)"
            ),
            "pca_falsification_rule": (
                "the 95% order-statistic interval lower endpoint "
                "for the tuned median residual exceeds the fixed "
                "threshold"
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
        "verdict": direct["verdict"] if passed else "BLOCKED",
        "passed": passed,
        "direct_checks": direct,
        "negative_control": negative_control,
    }
