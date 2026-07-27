from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

from sosmc_repro.claim1_checker import evaluate as evaluate_algorithm1
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
}
SMALL_BETA_SEEDS = [0]
BETA_VALUES = (0.25, 5.0)
TRUTH_GRID_LIMIT = 6.0
TRUTH_GRID_RESOLUTION = 400
TRUTH_GRID_BATCH = 65_536
TRUTH_GRID_VARIANTS = {
    "resolution_400_limit_6": (400, 6.0),
    "resolution_600_limit_6": (600, 6.0),
    "resolution_400_limit_8": (400, 8.0),
}


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _weight_summary(weights: Any) -> dict[str, float]:
    return {
        "sum": float(weights.sum().item()),
        "min": float(weights.min().item()),
        "max": float(weights.max().item()),
        "std": float(weights.std(unbiased=False).item()),
        "ess": float(1.0 / weights.square().sum().item()),
    }


def _install_algorithm1_trace(
    namespace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Trace the official EBM SOSMC loop and independently check its gradient."""
    torch = namespace["torch"]
    tuner_class = namespace["SOSMCULARewardTuner"]
    normalized_weights = namespace["normalized_weights_from_logA"]
    original_init = tuner_class.__init__
    original_step = tuner_class.step
    original_compute = tuner_class._compute_losses_on_xk
    original_propose = tuner_class._propose_and_alpha_forward
    original_resample = tuner_class._resample_if_needed
    registry: list[dict[str, Any]] = []

    def traced_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._orx_algorithm1_trace = {
            "implementation": (
                "official SOSMCULARewardTuner from the vendored authors' "
                "2D EBM notebook"
            ),
            "n_particles": int(self.cfg.n_particles),
            "outer_iterations_configured": int(self.cfg.n_outer_steps),
            "iterations": [],
            "gradient_checks": [],
        }
        self._orx_trace_entry = None
        self._orx_proposal_calls = 0
        registry.append(self._orx_algorithm1_trace)

    def traced_propose(
        self: Any, x_old: Any, gamma_k: float
    ) -> tuple[Any, Any]:
        self._orx_proposal_calls += 1
        return original_propose(self, x_old, gamma_k)

    def traced_compute(
        self: Any, x_model: Any, w: Any
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        loss_total, loss_rew, loss_kl, logs = original_compute(
            self, x_model, w
        )
        entry = self._orx_trace_entry
        if entry is None:
            return loss_total, loss_rew, loss_kl, logs

        params = [p for p in self.energy.parameters() if p.requires_grad]
        actual = torch.autograd.grad(
            loss_total,
            params,
            retain_graph=True,
            allow_unused=False,
        )
        x_independent = x_model.detach()
        w_independent = w.detach().view(-1)
        reward = self.reward_fn(x_independent).detach().view(-1)
        energy = self.energy(x_independent).view(-1)
        with torch.no_grad():
            energy_ref = self.energy_ref(x_independent).view(-1)
            delta = energy.detach() - energy_ref
            centered_delta = delta - (w_independent * delta).sum()
            centered_reward = reward - (w_independent * reward).sum()
            coefficients = w_independent * (
                centered_reward
                + float(self.cfg.beta_kl) * centered_delta
            )
        independently_reconstructed_loss = (
            coefficients.detach() * energy
        ).sum()
        independent = torch.autograd.grad(
            independently_reconstructed_loss,
            params,
            retain_graph=False,
            allow_unused=False,
        )
        actual_flat = torch.cat([value.reshape(-1) for value in actual])
        independent_flat = torch.cat(
            [value.reshape(-1) for value in independent]
        )
        difference = actual_flat - independent_flat
        relative_l2 = (
            difference.norm()
            / actual_flat.norm().clamp_min(
                torch.finfo(actual_flat.dtype).eps
            )
        )
        self._orx_algorithm1_trace["gradient_checks"].append(
            {
                "outer_iteration": int(entry["outer_iteration"]),
                "parameter_count": int(actual_flat.numel()),
                "actual_gradient_l2": float(actual_flat.norm().item()),
                "independent_gradient_l2": float(
                    independent_flat.norm().item()
                ),
                "relative_l2_error": float(relative_l2.item()),
                "max_absolute_error": float(
                    difference.abs().max().item()
                ),
                "independent_formula": (
                    "sum_i w_i * ((r_i-E_w[r]) + "
                    "beta*(delta_i-E_w[delta])) * grad_theta E_i"
                ),
            }
        )
        return loss_total, loss_rew, loss_kl, logs

    def traced_resample(
        self: Any, x_new: Any, log_a_new: Any
    ) -> tuple[Any, Any, float]:
        entry = self._orx_trace_entry
        if entry is not None:
            weights = normalized_weights(log_a_new).detach()
            entry["candidate_weights"] = _weight_summary(weights)
        result = original_resample(self, x_new, log_a_new)
        if entry is not None:
            entry["resampled"] = bool(
                torch.count_nonzero(result[1]).item() == 0
                and torch.count_nonzero(log_a_new).item() > 0
            )
        return result

    def traced_step(self: Any, k: int) -> None:
        if int(k) >= 3:
            original_step(self, k)
            return
        pre_weights = normalized_weights(self.logA).detach()
        entry: dict[str, Any] = {
            "outer_iteration": int(k),
            "pre_particle_sha256": _tensor_sha256(self.particles),
            "pre_log_weight_sha256": _tensor_sha256(self.logA),
            "pre_weights": _weight_summary(pre_weights),
        }
        proposal_calls_before = self._orx_proposal_calls
        self._orx_trace_entry = entry
        original_step(self, k)
        self._orx_trace_entry = None
        entry["proposal_calls"] = (
            self._orx_proposal_calls - proposal_calls_before
        )
        entry["post_particle_sha256"] = _tensor_sha256(self.particles)
        entry["post_log_weight_sha256"] = _tensor_sha256(self.logA)
        self._orx_algorithm1_trace["iterations"].append(entry)

    tuner_class.__init__ = traced_init
    tuner_class._propose_and_alpha_forward = traced_propose
    tuner_class._compute_losses_on_xk = traced_compute
    tuner_class._resample_if_needed = traced_resample
    tuner_class.step = traced_step
    return registry


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
        # The trigger frequency is unchanged, but a method-independent 2D
        # grid quadrature installed below replaces stochastic evaluation MCMC.
        "n_eval_fresh": 500,
        "eval_n_samples": 1,
        "eval_langevin_steps": 1,
        "eval_thin": 1,
        "eval_burn_in": 0,
        "eval_step_size": 5e-3,
        "eval_noise_scale": 1.0,
        "eval_clamp_value": None,
        "seed": seed,
        "beta_kl": beta_kl,
    }


def _install_grid_truth_evaluator(namespace: dict[str, Any]) -> None:
    torch = namespace["torch"]
    batched_energy = namespace["_batched_energy"]

    @torch.no_grad()
    def integrate_grid(
        energy: Any,
        energy_ref: Any,
        reward_fn: Any,
        device: str,
        resolution: int,
        limit: float,
    ) -> dict[str, float]:
        axis = torch.linspace(
            -limit,
            limit,
            resolution,
            device=device,
        )
        xx, yy = torch.meshgrid(axis, axis, indexing="xy")
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        energy = batched_energy(
            energy, grid, batch=TRUTH_GRID_BATCH
        ).double()
        energy_ref = batched_energy(
            energy_ref, grid, batch=TRUTH_GRID_BATCH
        ).double()
        cell = ((2.0 * limit) / (resolution - 1)) ** 2
        log_z = torch.logsumexp(-energy, dim=0) + torch.log(
            torch.as_tensor(cell, dtype=torch.float64, device=device)
        )
        log_z_ref = torch.logsumexp(-energy_ref, dim=0) + torch.log(
            torch.as_tensor(cell, dtype=torch.float64, device=device)
        )
        log_p = -energy - log_z
        log_p_ref = -energy_ref - log_z_ref
        mass = log_p.exp() * cell
        reward = reward_fn(grid).reshape(-1).double()
        mean_reward = (mass * reward).sum()
        reverse_kl = (mass * (log_p - log_p_ref)).sum()
        return {
            "mean": float(mean_reward.item()),
            "kl_grid": float(reverse_kl.item()),
            "logZ": float(log_z.item()),
            "logZ0": float(log_z_ref.item()),
            "resolution": resolution,
            "limit": limit,
        }

    @torch.no_grad()
    def evaluate_grid(self: Any) -> dict[str, float]:
        """Integrate reward and reverse KL under the normalized 2D EBM."""
        self.energy.eval()
        variants = {
            name: integrate_grid(
                self.energy,
                self.energy_ref,
                self.reward_fn,
                self.device,
                resolution,
                limit,
            )
            for name, (resolution, limit) in TRUTH_GRID_VARIANTS.items()
        }
        self.history.setdefault("truth_grid_sensitivity", []).append(variants)
        self.energy.train()
        return variants["resolution_400_limit_6"]

    namespace["_sosmc_integrate_grid"] = integrate_grid
    namespace["IDRewardTuner"]._eval_fresh = evaluate_grid
    namespace["SOSMCULARewardTuner"]._eval_fresh = evaluate_grid


def _install_paired_reference_particle_cache(
    namespace: dict[str, Any],
) -> dict[str, Any]:
    """Reuse one exact reference draw only for an identical paired setup."""
    torch = namespace["torch"]
    original = namespace["generate_langevin_samples_from_energy"]
    cached_particles = None
    cached_state = None
    cached_sampler_config = None
    stats: dict[str, Any] = {
        "cache_misses": 0,
        "cache_hits": 0,
        "reference_parameters_bitwise_equal": None,
        "sampler_configuration_equal": None,
    }
    sampler_fields = (
        "n_samples",
        "n_steps",
        "step_size",
        "noise_scale",
        "clamp_value",
        "particle_init_lim",
        "device",
    )

    def paired_generator(*args: Any, **kwargs: Any) -> Any:
        nonlocal cached_particles, cached_state, cached_sampler_config
        is_reference_initialization = (
            not args
            and int(kwargs.get("n_samples", -1)) == 10_000
            and int(kwargs.get("n_steps", -1)) == 20_000
        )
        if not is_reference_initialization:
            return original(*args, **kwargs)

        sampler_config = {
            field: kwargs.get(field) for field in sampler_fields
        }
        model = kwargs["energy_model"]
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        if cached_particles is None:
            stats["cache_misses"] += 1
            cached_particles = original(*args, **kwargs).detach().clone()
            cached_state = state
            cached_sampler_config = sampler_config
            return cached_particles.clone()

        sampler_equal = sampler_config == cached_sampler_config
        state_equal = (
            state.keys() == cached_state.keys()
            and all(
                torch.equal(state[name], cached_state[name])
                for name in state
            )
        )
        stats["sampler_configuration_equal"] = sampler_equal
        stats["reference_parameters_bitwise_equal"] = state_equal
        if not sampler_equal or not state_equal:
            raise RuntimeError(
                "Refusing paired-particle reuse: reference model or sampler "
                "configuration differs."
            )
        stats["cache_hits"] += 1
        return cached_particles.clone()

    namespace["generate_langevin_samples_from_energy"] = paired_generator
    return stats


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
        grid_sensitivity = {
            name: {
                **values,
                "objective": float(values["mean"])
                - beta_kl * float(values["kl_grid"]),
            }
            for name, values in history["truth_grid_sensitivity"][index].items()
        }
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
                "truth_grid_sensitivity": grid_sensitivity,
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
    _install_grid_truth_evaluator(namespace)
    reference_cache = _install_paired_reference_particle_cache(namespace)
    algorithm1_registry = _install_algorithm1_trace(namespace)
    run_trial = namespace["run_experimental_trial"]
    reward_fn = namespace["reward_lower_halfplane"]
    rows: list[dict[str, Any]] = []
    trial_metadata: list[dict[str, Any]] = []

    previous_cwd = Path.cwd()
    os.chdir(NOTEBOOK_DIR)
    try:
        specifications = [
            (dataset, alias, seed, beta_kl)
            for dataset, alias in DATASETS.items()
            for seed in SMALL_BETA_SEEDS
            for beta_kl in BETA_VALUES
        ]
        for dataset, alias, seed, beta_kl in specifications:
            trial_started = time.perf_counter()
            config = _trial_config(reward_fn, alias, seed, beta_kl)
            result = run_trial(config, run_impdiff=True, run_sosmc=True)
            p0_grid = namespace["_sosmc_integrate_grid"](
                result["energy_ref"],
                result["energy_ref"],
                reward_fn,
                "cpu",
                TRUTH_GRID_RESOLUTION,
                TRUTH_GRID_LIMIT,
            )
            result["p0A"] = float(p0_grid["mean"])
            result["opt_reward"] = float(
                namespace["optimal_indicator_reward"](
                    result["p0A"], beta_kl
                )
            )
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
    if len(algorithm1_registry) != len(trial_metadata):
        raise RuntimeError(
            "Expected one official SOSMC trace for every 2D EBM trial."
        )
    algorithm1_traces = []
    for trace in algorithm1_registry:
        trace["reference_initialization"] = reference_cache
        trace["official_notebook_sha256"] = (
            "8b3938b65467238b07860caa071b7f3cb48eb5a77aab1a0292a32a0ee599c514"
        )
        trace["upstream_commit"] = (
            "62e4f8f07ae2705073388f5d2c4babf5c87b00be"
        )
        algorithm1_traces.append(
            {
                "raw_trace": trace,
                "independent_checker": evaluate_algorithm1(trace),
            }
        )
    algorithm1_passed = all(
        trace["independent_checker"]["passed"]
        for trace in algorithm1_traces
    )
    algorithm1_result = {
        "claim": "Section 3.2 Algorithm 1 on the official 2D EBM",
        "verdict": "VERIFIED" if algorithm1_passed else "BLOCKED",
        "passed": algorithm1_passed,
        "trials": algorithm1_traces,
        "negative_controls": [
            trace["independent_checker"]["negative_control"]
            for trace in algorithm1_traces
        ],
    }
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
            "large_beta_control": 5.0,
            "n_particles": 10_000,
            "n_outer_steps": 1_001,
            "fresh_eval_frequency": 500,
            "truth_evaluator": "normalized dense-grid quadrature",
            "truth_grid_limit": TRUTH_GRID_LIMIT,
            "truth_grid_resolution": TRUTH_GRID_RESOLUTION,
            "truth_grid_batch": TRUTH_GRID_BATCH,
            "truth_grid_variants": TRUTH_GRID_VARIANTS,
            "fresh_eval_sampling_error": 0.0,
            "paired_reference_particle_cache": reference_cache,
            "checkpoint_device_override": "cpu",
        },
        "trial_metadata": trial_metadata,
        "raw_rows": rows,
        "independent_checker": checker,
        "algorithm1_result": algorithm1_result,
        "runtime_seconds": time.perf_counter() - started,
        "passed": checker["passed"],
    }
