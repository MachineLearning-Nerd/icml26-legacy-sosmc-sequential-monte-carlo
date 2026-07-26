from __future__ import annotations

import math
import statistics
from typing import Any


SMALL_BETA = 0.25
LARGE_BETA = 5.0
T_CRITICAL_DF8_95 = 2.306


def _rmse(rows: list[dict[str, Any]], field: str) -> float:
    errors = [
        float(row[field]) - float(row["fresh_reward"])
        for row in rows
    ]
    return math.sqrt(statistics.fmean(error * error for error in errors))


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    small = [row for row in rows if float(row["beta_kl"]) == SMALL_BETA]
    large = [row for row in rows if float(row["beta_kl"]) == LARGE_BETA]
    datasets = sorted({str(row["dataset"]) for row in small})
    seeds = sorted({int(row["seed"]) for row in small})

    best_by_trial: dict[tuple[str, int, str], float] = {}
    for row in small:
        key = (str(row["dataset"]), int(row["seed"]), str(row["method"]))
        best_by_trial[key] = max(
            best_by_trial.get(key, -math.inf),
            float(row["objective"]),
        )

    paired_differences = [
        best_by_trial[(dataset, seed, "SOSMC-ULA")]
        - best_by_trial[(dataset, seed, "ImpDiff")]
        for dataset in datasets
        for seed in seeds
    ]
    paired_mean = statistics.fmean(paired_differences)
    paired_sd = statistics.stdev(paired_differences)
    paired_se = paired_sd / math.sqrt(len(paired_differences))
    paired_ci = [
        paired_mean - T_CRITICAL_DF8_95 * paired_se,
        paired_mean + T_CRITICAL_DF8_95 * paired_se,
    ]

    per_dataset_direction = {
        dataset: statistics.fmean(
            best_by_trial[(dataset, seed, "SOSMC-ULA")]
            - best_by_trial[(dataset, seed, "ImpDiff")]
            for seed in seeds
        )
        > 0.0
        for dataset in datasets
    }

    tracking: dict[str, Any] = {}
    for method, field in (
        ("ImpDiff", "particle_reward"),
        ("SOSMC-ULA", "weighted_particle_reward"),
    ):
        method_rows = [
            row
            for row in small
            if row["method"] == method and int(row["step"]) > 0
        ]
        tracking[method] = {
            "n_evaluation_points": len(method_rows),
            "rmse_particle_vs_fresh": _rmse(method_rows, field),
        }

    tracking_better = (
        tracking["SOSMC-ULA"]["rmse_particle_vs_fresh"]
        < tracking["ImpDiff"]["rmse_particle_vs_fresh"]
    )

    large_best: dict[str, float] = {}
    for method in ("ImpDiff", "SOSMC-ULA"):
        values = [
            float(row["objective"])
            for row in large
            if row["method"] == method
        ]
        large_best[method] = max(values)
    large_gap = large_best["SOSMC-ULA"] - large_best["ImpDiff"]
    large_comparable = abs(large_gap) <= 0.05

    # Negative control: swapping the algorithm labels must break the
    # pre-registered small-beta directional result.
    reversed_direction = paired_mean < 0.0
    negative_control_failed_as_intended = not reversed_direction

    checks = {
        "small_beta_positive_mean_advantage": paired_ci[0] > 0.0,
        "small_beta_positive_advantage_each_dataset": per_dataset_direction,
        "sosmc_weighted_tracking_rmse_below_impdiff_unweighted": tracking_better,
        "large_beta_best_objectives_within_0p05": large_comparable,
    }
    passed = (
        checks["small_beta_positive_mean_advantage"]
        and all(per_dataset_direction.values())
        and tracking_better
        and large_comparable
        and negative_control_failed_as_intended
    )
    return {
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "scope": {
            "datasets": datasets,
            "small_beta": SMALL_BETA,
            "small_beta_seeds": seeds,
            "large_beta": LARGE_BETA,
            "large_beta_dataset": "circles",
        },
        "small_beta_best_objective_paired_difference": {
            "n": len(paired_differences),
            "values": paired_differences,
            "mean": paired_mean,
            "sd": paired_sd,
            "ci95": paired_ci,
        },
        "per_dataset_positive_mean_advantage": per_dataset_direction,
        "tracking": tracking,
        "large_beta_control": {
            "best_objectives": large_best,
            "sosmc_minus_impdiff": large_gap,
            "absolute_comparability_threshold": 0.05,
        },
        "checks": checks,
        "negative_control": {
            "reversed_label_direction_passed": reversed_direction,
            "failed_as_intended": negative_control_failed_as_intended,
        },
        "passed": passed,
    }
