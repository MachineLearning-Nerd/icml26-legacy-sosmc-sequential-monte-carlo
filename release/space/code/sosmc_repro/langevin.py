from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from sosmc_repro.claim4_checker import evaluate
from sosmc_repro.io import ROOT
from sosmc_repro.notebook_loader import execute_cells


NOTEBOOK = (
    ROOT
    / "vendor"
    / "SOSMC"
    / "reward_tuning"
    / "langevin_processes"
    / "experiments.ipynb"
)
DEFINITION_CELLS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

PROBLEMS = [
    {
        "name": "dual_smooth",
        "dist_key": "dual_gaussian_sig1",
        "reward_key": "smooth_right",
        "beta_kl": 0.1,
        "paper_sosmc_mean": 0.6484,
        "paper_impdiff_mean": 0.6026,
        "paper_soul_mean": 0.6477,
    },
    {
        "name": "dual_hard",
        "dist_key": "dual_gaussian_sig1",
        "reward_key": "hard_right",
        "beta_kl": 0.001,
        "paper_sosmc_mean": 0.0186,
        "paper_impdiff_mean": 0.0169,
        "paper_soul_mean": 0.0187,
    },
    {
        "name": "sparse_hard",
        "dist_key": "hex_sparse_sig2",
        "reward_key": "hard_right",
        "beta_kl": 0.001,
        "paper_sosmc_mean": 0.0369,
        "paper_impdiff_mean": 0.0363,
        "paper_soul_mean": 0.0365,
    },
    {
        "name": "tight_tight",
        "dist_key": "hex_tight_sig002",
        "reward_key": "hard_right_tight",
        "beta_kl": 0.1,
        "paper_sosmc_mean": 0.2497,
        "paper_impdiff_mean": 0.0142,
        "paper_soul_mean": 0.0,
    },
]
METHODS = ["ImpDiff_Adam", "SOUL_Adam", "SOSMC-ULA_Adam"]


def run_wallclock() -> dict[str, Any]:
    started = time.perf_counter()
    namespace = execute_cells(NOTEBOOK, DEFINITION_CELLS)
    run_sweep = namespace["run_sweep"]
    rows: list[dict[str, Any]] = []

    for problem in PROBLEMS:
        sweep = run_sweep(
            dist_key=problem["dist_key"],
            reward_key=problem["reward_key"],
            loop_types=METHODS,
            seeds=list(range(10)),
            beta_kls=[problem["beta_kl"]],
            run_settings={
                "run_mode": "time",
                "max_time_seconds": 2.001,
                "max_steps": None,
                "utilise_warmup_step": True,
            },
            base_config=namespace["CONFIG_BASE"],
            config_reward=namespace["CONFIG_REWARD"],
            config_distribution=namespace["CONFIG_DISTRIBUTION"],
            config_loop=namespace["CONFIG_LOOP"],
            save_dir=None,
            force_rerun=True,
            verbose=False,
        )
        for method in METHODS:
            for seed, history in enumerate(
                sweep[(method, float(problem["beta_kl"]))]
            ):
                reward = np.asarray(history["reward"], dtype=float)
                times = np.asarray(history["time"], dtype=float)
                rows.append(
                    {
                        "problem": problem["name"],
                        "method": method,
                        "seed": seed,
                        "beta_kl": problem["beta_kl"],
                        "terminal_reward": float(reward[-1]),
                        "steps_completed": int(reward.size),
                        "last_recorded_time_seconds": float(times[-1]),
                    }
                )

    checker = evaluate(rows)
    return {
        "claim": "Section 5.1 wall-clock Langevin reward tuning",
        "verdict": checker["verdict"],
        "official_notebook_sha256": "cf0467361311b1b03786a4eef8d10e64890d2a02afc6f4d649fec42f05b25b89",
        "upstream_commit": "62e4f8f07ae2705073388f5d2c4babf5c87b00be",
        "paper_table_2_targets": PROBLEMS,
        "raw_rows": rows,
        "independent_checker": checker,
        "runtime_seconds": time.perf_counter() - started,
        "passed": checker["passed"],
    }
