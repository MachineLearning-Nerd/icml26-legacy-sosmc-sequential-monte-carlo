from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from sosmc_repro.claim5_checker import evaluate
from sosmc_repro.io import ROOT
from sosmc_repro.notebook_loader import execute_cells


NOTEBOOK = (
    ROOT
    / "vendor"
    / "SOSMC"
    / "reward_tuning"
    / "ebms_2D"
    / "experiments.ipynb"
)
NOTEBOOK_DIR = NOTEBOOK.parent
DEFINITION_CELLS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
DATASETS = {
    "circles": "ebm_circles",
    "two_moons": "ebm_two_moons",
    "blobs": "ebm_blobs",
}
SMALL_BETA_SEEDS = [0]


def _trial_config(
    reward_fn: Any,
    dataset_alias: str,
    seed: int,
    beta_kl: float,
) -> dict[str, Any]:
    return {
        "dataset_alias": dataset_alias,
        "checkpoint": "latest",
        "plot_n_samples": 0,
        "plot_langevin_steps": 1,
        "plot_lim": 6.0,
        "plot_every": 10**9,
        "log_every": 10**9,
        "lr": 2e-4,
        "particle_reinit_prob": 0,
        "optimiser_alias": "adam",
        "optimiser_kwargs": None,
        "n_particles": 10_000,
        "n_outer_steps": 1_001,
        "reward_fn": reward_fn,
        "log_detailed_stats": False,
        "log_kl_estimates": True,
        "sampler_steps_per_outer": 1,
        "gamma_impdiff": 5e-3,
        "noise_scale_impdiff": 1.0,
        "clamp_value_impdiff": None,
        "gamma_sosmc": 5e-3,
        "gamma_sosmc_max": 1e-2,
        "gamma_sosmc_min": 1e-8,
        "adapt_factor": 1.01,
        "noise_scale_sosmc": 1.0,
        "ess_resample_ratio": 0.9,
        "ess_adapt_ratio": 0.95,
        # The official notebook's documented reduced-evaluation setting uses
        # a 5,000-step chain with 500 burn-in steps. Start/mid/end evaluations
        # retain trajectory-wide information within the CPU job cap.
        "n_eval_fresh": 500,
        "eval_n_samples": 1_000,
        "eval_langevin_steps": 5_000,
        "eval_thin": 1,
        "eval_burn_in": 500,
        "eval_step_size": 5e-3,
        "eval_noise_scale": 1.0,
        "eval_clamp_value": None,
        "seed": seed,
        "beta_kl": beta_kl,
    }


def _rows(
    history: dict[str, Any],
    dataset: str,
    seed: int,
    beta_kl: float,
    method: str,
) -> list[dict[str, Any]]:
    step_to_index = {
        int(step): index for index, step in enumerate(history["step"])
    }
    particle_key = (
        "mean_reward_weighted" if method == "SOSMC-ULA" else "mean_reward"
    )
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(history["fresh_eval_step"]):
        outer_index = step_to_index[int(step)]
        particle_reward = float(history["mean_reward"][outer_index])
        weighted_particle_reward = float(history[particle_key][outer_index])
        fresh_reward = float(history["fresh_reward_mean"][index])
        fresh_kl = float(history["fresh_kl_grid"][index])
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "beta_kl": beta_kl,
                "method": method,
                "step": int(step),
                "fresh_reward": fresh_reward,
                "fresh_kl_grid": fresh_kl,
                "objective": fresh_reward - beta_kl * fresh_kl,
                "particle_reward": particle_reward,
                "weighted_particle_reward": weighted_particle_reward,
            }
        )
    return rows


def run_2d_suite() -> dict[str, Any]:
    started = time.perf_counter()
    namespace = execute_cells(NOTEBOOK, DEFINITION_CELLS)
    official_load_trainer = namespace["load_trainer"]

    def load_trainer_cpu(
        root_dir: str | Path,
        experiment_name: str,
        checkpoint: str | int | Path = "latest",
        device: str | None = None,
    ) -> Any:
        del device
        return official_load_trainer(
            root_dir,
            experiment_name,
            checkpoint=checkpoint,
            device="cpu",
        )

    # The supplied checkpoint configs store the authors' original CUDA device.
    # Use their loader's documented device override to enforce this campaign's
    # CPU-only compute contract without altering checkpoint content.
    namespace["load_trainer"] = load_trainer_cpu
    run_trial = namespace["run_experimental_trial"]
    reward_fn = namespace["reward_lower_halfplane"]
    rows: list[dict[str, Any]] = []
    trial_metadata: list[dict[str, Any]] = []

    previous_cwd = Path.cwd()
    os.chdir(NOTEBOOK_DIR)
    try:
        specifications = [
            (dataset, alias, seed, 0.25)
            for dataset, alias in DATASETS.items()
            for seed in SMALL_BETA_SEEDS
        ]
        for dataset, alias, seed, beta_kl in specifications:
            trial_started = time.perf_counter()
            config = _trial_config(reward_fn, alias, seed, beta_kl)
            result = run_trial(config, run_impdiff=True, run_sosmc=True)
            rows.extend(
                _rows(
                    result["history_impdiff"],
                    dataset,
                    seed,
                    beta_kl,
                    "ImpDiff",
                )
            )
            rows.extend(
                _rows(
                    result["history_sosmc"],
                    dataset,
                    seed,
                    beta_kl,
                    "SOSMC-ULA",
                )
            )
            trial_metadata.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "beta_kl": beta_kl,
                    "p0_reward_mass": float(result["p0A"]),
                    "analytic_optimal_reward": float(result["opt_reward"]),
                    "runtime_seconds": time.perf_counter() - trial_started,
                }
            )
    finally:
        os.chdir(previous_cwd)

    checker = evaluate(rows)
    return {
        "claim": "Section 5.2 checkpointed 2D EBM reward tuning",
        "verdict": checker["verdict"],
        "official_notebook_sha256": "8b3938b65467238b07860caa071b7f3cb48eb5a77aab1a0292a32a0ee599c514",
        "upstream_commit": "62e4f8f07ae2705073388f5d2c4babf5c87b00be",
        "configuration": {
            "datasets": DATASETS,
            "reward": "lower_halfplane",
            "small_beta": 0.25,
            "small_beta_seeds": SMALL_BETA_SEEDS,
            "large_beta_control": "not run in this runtime-bounded shard",
            "n_particles": 10_000,
            "n_outer_steps": 1_001,
            "fresh_eval_frequency": 500,
            "fresh_eval_samples": 1_000,
            "fresh_eval_langevin_steps": 5_000,
            "fresh_eval_burn_in": 500,
            "checkpoint_device_override": "cpu",
        },
        "trial_metadata": trial_metadata,
        "raw_rows": rows,
        "independent_checker": checker,
        "runtime_seconds": time.perf_counter() - started,
        "passed": checker["passed"],
    }
