from __future__ import annotations

import math
import statistics
from typing import Any


METHODS = ("ImpDiff_Adam", "SOUL_Adam", "SOSMC-ULA_Adam")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["problem"], row["method"]), []).append(
            float(row["terminal_reward"])
        )

    summaries: dict[str, Any] = {}
    problems = sorted({row["problem"] for row in rows})
    for problem in problems:
        summaries[problem] = {}
        for method in METHODS:
            values = grouped[(problem, method)]
            mean = statistics.fmean(values)
            sd = statistics.stdev(values)
            se = sd / math.sqrt(len(values))
            summaries[problem][method] = {
                "n": len(values),
                "mean": mean,
                "sd": sd,
                "ci95": [mean - 2.262 * se, mean + 2.262 * se],
                "values": values,
            }

        imp = summaries[problem]["ImpDiff_Adam"]["values"]
        sos = summaries[problem]["SOSMC-ULA_Adam"]["values"]
        paired = [s - i for s, i in zip(sos, imp)]
        mean_diff = statistics.fmean(paired)
        sd_diff = statistics.stdev(paired)
        se_diff = sd_diff / math.sqrt(len(paired))
        summaries[problem]["paired_sosmc_minus_impdiff"] = {
            "mean": mean_diff,
            "sd": sd_diff,
            "ci95": [mean_diff - 2.262 * se_diff, mean_diff + 2.262 * se_diff],
        }
    return summaries


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = summarize(rows)
    direction_checks = {
        problem: data["SOSMC-ULA_Adam"]["mean"] > data["ImpDiff_Adam"]["mean"]
        for problem, data in summaries.items()
    }
    smooth_variance_check = (
        summaries["dual_smooth"]["SOSMC-ULA_Adam"]["sd"]
        < summaries["dual_smooth"]["SOUL_Adam"]["sd"]
    )
    tight_failure_check = (
        summaries["tight_tight"]["SOSMC-ULA_Adam"]["mean"]
        > summaries["tight_tight"]["SOUL_Adam"]["mean"] + 0.05
    )

    # Checker negative control: reversing the claimed labels must not satisfy
    # the all-setting superiority contract.
    reversed_checks = {
        problem: data["ImpDiff_Adam"]["mean"] > data["SOSMC-ULA_Adam"]["mean"]
        for problem, data in summaries.items()
    }
    negative_control_failed_as_intended = not all(reversed_checks.values())

    passed = (
        all(direction_checks.values())
        and smooth_variance_check
        and tight_failure_check
        and negative_control_failed_as_intended
    )
    return {
        "verdict": "VERIFIED" if passed else "BLOCKED",
        "summaries": summaries,
        "checks": {
            "sosmc_mean_exceeds_impdiff_all_four_settings": direction_checks,
            "figure_1a_sosmc_sd_below_soul": smooth_variance_check,
            "figure_1b_soul_mode_failure_not_seen_for_sosmc": tight_failure_check,
        },
        "negative_control": {
            "reversed_label_checks": reversed_checks,
            "failed_as_intended": negative_control_failed_as_intended,
        },
        "passed": passed,
    }

