from __future__ import annotations

import copy
import gc
import math
import random
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sosmc_repro.claim6_checker import evaluate
from sosmc_repro.io import ROOT, sha256_file
from sosmc_repro.notebook_loader import execute_cells


NOTEBOOK = (
    ROOT
    / "vendor"
    / "SOSMC"
    / "reward_tuning"
    / "ebms_mnist"
    / "experiments.ipynb"
)
CHECKPOINT = (
    NOTEBOOK.parent / "saved_models" / "tutorial8" / "MNIST.ckpt"
)
DEFINITION_CELLS = [4, 8, 10, 16, 18]
REWARD_NAMES = ("bright", "dark", "lower_half")
BETAS = (5.0, 2.0, 1.0, 0.5)
TRAIN_SEED = 42
OUTER_NOISE_SEED = 2026072701
EVALUATION_SEEDS = (2026072711, 2026072712, 2026072713)
N_PARTICLES = 1_000
N_OUTER = 1_000
TUNING_GAMMA = 5e-3
TUTORIAL_INITIALIZATION_STEPS = 1_000
EVAL_SAMPLES_PER_SEED = 64
EVAL_STEPS = 512
CLASSIFIER_SEED = 2026072721
CLASSIFIER_EPOCHS = 3
MNIST_CACHE = ROOT / ".openresearch" / "cache" / "mnist"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _state_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[name], right[name]) for name in left
    )


def _install_initial_particle_cache(
    namespace: dict[str, Any],
) -> dict[str, Any]:
    original = namespace["init_persistent_particles_pi0"]
    cached_particles = None
    cached_state = None
    cached_config = None
    stats: dict[str, Any] = {
        "cache_misses": 0,
        "cache_hits": 0,
        "all_reference_parameters_bitwise_equal": True,
        "all_sampler_configurations_equal": True,
    }
    config_fields = (
        "n_particles",
        "device",
        "tutorial_steps",
        "tutorial_step_size",
        "purify_steps",
        "gamma",
        "noise_scale",
        "clamp_value",
    )

    def cached(**kwargs: Any) -> torch.Tensor:
        nonlocal cached_particles, cached_state, cached_config
        state = {
            name: value.detach().cpu().clone()
            for name, value in kwargs["energy_ref"].state_dict().items()
        }
        config = {
            field: str(kwargs.get(field))
            if field == "device"
            else kwargs.get(field)
            for field in config_fields
        }
        if cached_particles is None:
            stats["cache_misses"] += 1
            cached_particles = original(**kwargs).detach().cpu().clone()
            cached_state = state
            cached_config = config
            return cached_particles.to(kwargs["device"]).clone()
        state_matches = _state_equal(state, cached_state)
        config_matches = config == cached_config
        stats["all_reference_parameters_bitwise_equal"] &= state_matches
        stats["all_sampler_configurations_equal"] &= config_matches
        if not state_matches or not config_matches:
            raise RuntimeError(
                "Refusing MNIST particle reuse: reference model or sampler "
                "configuration differs."
            )
        stats["cache_hits"] += 1
        return cached_particles.to(kwargs["device"]).clone()

    namespace["init_persistent_particles_pi0"] = cached
    return stats


class DigitRecognizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def _mnist_tensor(dataset: Any, indices: torch.Tensor) -> torch.Tensor:
    return (
        dataset.data[indices].float().unsqueeze(1) / 127.5 - 1.0
    )


def _balanced_indices(
    targets: torch.Tensor, per_class: int
) -> torch.Tensor:
    parts = []
    for label in range(10):
        parts.append(
            torch.nonzero(targets == label, as_tuple=False)
            .view(-1)[:per_class]
        )
    return torch.cat(parts)


@torch.no_grad()
def _recognizer_features(
    model: DigitRecognizer, samples: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits = []
    all_features = []
    for start in range(0, samples.shape[0], 256):
        logits, features = model(
            samples[start : start + 256], return_features=True
        )
        all_logits.append(logits)
        all_features.append(features)
    return torch.cat(all_logits), torch.cat(all_features)


def _train_recognizer() -> dict[str, Any]:
    started = time.perf_counter()
    _seed_all(CLASSIFIER_SEED)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_data = datasets.MNIST(
        MNIST_CACHE, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        MNIST_CACHE, train=False, download=True, transform=transform
    )
    generator = torch.Generator().manual_seed(CLASSIFIER_SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )
    model = DigitRecognizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_losses = []
    model.train()
    for _ in range(CLASSIFIER_EPOCHS):
        loss_sum = 0.0
        count = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * images.shape[0]
            count += images.shape[0]
        epoch_losses.append(loss_sum / count)

    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            prediction = model(images).argmax(dim=1)
            correct += int((prediction == labels).sum().item())
            count += labels.numel()

    reference_indices = _balanced_indices(train_data.targets, 100)
    real_test_indices = _balanced_indices(test_data.targets, 100)
    reference_images = _mnist_tensor(train_data, reference_indices)
    real_test_images = _mnist_tensor(test_data, real_test_indices)
    _, reference_features = _recognizer_features(
        model, reference_images
    )
    _, real_test_features = _recognizer_features(model, real_test_images)
    feature_mean = reference_features.mean(dim=0)
    feature_std = reference_features.std(dim=0, unbiased=False).clamp_min(
        1e-6
    )
    reference_standardized = (
        reference_features - feature_mean
    ) / feature_std
    real_standardized = (real_test_features - feature_mean) / feature_std
    real_distances = _nearest_support_distance(
        real_standardized, reference_standardized
    )
    dataset_files = {
        str(path.relative_to(MNIST_CACHE)): sha256_file(path)
        for path in sorted(MNIST_CACHE.rglob("*"))
        if path.is_file()
    }
    return {
        "model": model,
        "reference_features": reference_standardized,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "test_accuracy": correct / count,
        "epoch_losses": epoch_losses,
        "real_test_support_distance": _distribution_summary(
            real_distances
        ),
        "dataset_file_sha256": dataset_files,
        "training_seed": CLASSIFIER_SEED,
        "training_epochs": CLASSIFIER_EPOCHS,
        "runtime_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def _nearest_support_distance(
    samples: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    parts = []
    scale = math.sqrt(reference.shape[1])
    for start in range(0, samples.shape[0], 256):
        distances = torch.cdist(
            samples[start : start + 256], reference
        )
        parts.append(distances.min(dim=1).values / scale)
    return torch.cat(parts)


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().view(-1)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=True).item()),
        "median": float(values.median().item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


@torch.no_grad()
def _digit_diagnostics(
    recognizer: dict[str, Any],
    samples: torch.Tensor,
) -> dict[str, float]:
    model = recognizer["model"]
    logits, features = _recognizer_features(model, samples)
    probabilities = logits.softmax(dim=1)
    confidence = probabilities.max(dim=1).values
    standardized = (
        features - recognizer["feature_mean"]
    ) / recognizer["feature_std"]
    support = _nearest_support_distance(
        standardized, recognizer["reference_features"]
    )
    return {
        "classifier_confidence_mean": float(confidence.mean().item()),
        "classifier_confidence_median": float(confidence.median().item()),
        "support_distance_mean": float(support.mean().item()),
        "support_distance_median": float(support.median().item()),
        "support_distance_q95": float(
            torch.quantile(support.double(), 0.95).item()
        ),
        "predicted_class_count": int(
            probabilities.argmax(dim=1).unique().numel()
        ),
    }


def _paired_interval(
    baseline: torch.Tensor, tuned: torch.Tensor
) -> dict[str, float]:
    difference = (tuned - baseline).detach().double().view(-1)
    standard_deviation = difference.std(unbiased=True)
    standard_error = standard_deviation / math.sqrt(difference.numel())
    half_width = 1.96 * standard_error
    mean = difference.mean()
    return {
        "count": int(difference.numel()),
        "mean": float(mean.item()),
        "std": float(standard_deviation.item()),
        "standard_error": float(standard_error.item()),
        "ci95_low": float((mean - half_width).item()),
        "ci95_high": float((mean + half_width).item()),
    }


def _reward_summary(values: torch.Tensor) -> dict[str, float]:
    summary = _distribution_summary(values)
    return {
        "count": summary["count"],
        "mean": summary["mean"],
        "std": summary["std"],
        "minimum": summary["minimum"],
        "maximum": summary["maximum"],
    }


def _reward_functions(
    namespace: dict[str, Any],
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    def lower_half(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1.0, 1.0)
        midpoint = x.shape[-2] // 2
        top = x[..., :midpoint, :]
        bottom = x[..., midpoint:, :]
        return 0.5 * (
            bottom.mean(dim=(1, 2, 3))
            - top.mean(dim=(1, 2, 3))
        )

    return {
        "bright": namespace["reward_bright"],
        "dark": namespace["reward_dark"],
        "lower_half": lower_half,
    }


def _hack_samples(reward: str, count: int) -> torch.Tensor:
    if reward == "bright":
        return torch.ones(count, 1, 28, 28)
    if reward == "dark":
        return -torch.ones(count, 1, 28, 28)
    samples = -torch.ones(count, 1, 28, 28)
    samples[:, :, 14:, :] = 1.0
    return samples


def run_mnist_suite() -> dict[str, Any]:
    started = time.perf_counter()
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only campaign refuses a visible CUDA device.")
    namespace = execute_cells(NOTEBOOK, DEFINITION_CELLS)
    namespace["CHECKPOINT_PATH"] = str(CHECKPOINT.parent)
    namespace["device"] = torch.device("cpu")
    initial_cache = _install_initial_particle_cache(namespace)
    recognizer = _train_recognizer()
    rewards = _reward_functions(namespace)
    load_model = namespace["load_pretrained_dem"]
    energy_class = namespace["MNISTEnergy"]
    tuner_class = namespace["SOSMCULATuner"]
    sample_original = namespace[
        "sample_with_original_tutorial_sampler"
    ]

    tuned_states: dict[tuple[str, float], dict[str, torch.Tensor]] = {}
    training_diagnostics = []
    for reward_name in REWARD_NAMES:
        reward_fn = rewards[reward_name]
        for beta in BETAS:
            trial_started = time.perf_counter()
            _seed_all(TRAIN_SEED)
            model = load_model().to("cpu")
            model.eval()
            energy = energy_class(model.cnn).to("cpu")
            reference = copy.deepcopy(energy).to("cpu").eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            tuner = tuner_class(
                energy_model=energy,
                reward_fn=reward_fn,
                device=torch.device("cpu"),
                n_particles=N_PARTICLES,
                lr=1e-4,
                beta_kl=beta,
                gamma=TUNING_GAMMA,
                noise_scale=1.0,
                grad_clip_norm=1e8,
                energy_reference=reference,
                ess_resample_ratio=0.9,
                ess_adapt_ratio=0.95,
                gamma_min=1e-8,
                gamma_max=1e-2,
                adapt_factor=1.0,
                tutorial_steps=TUTORIAL_INITIALIZATION_STEPS,
                purify_steps=0,
                eval_sampler_fn=None,
                eval_fresh_frequency=0,
                eval_fresh_n_samples=EVAL_SAMPLES_PER_SEED,
                eval_fresh_n_steps=EVAL_STEPS,
                eval_fresh_step_size=10.0,
            )
            _seed_all(OUTER_NOISE_SEED)
            history = tuner.run(N_OUTER, log_every=250)
            tuned_states[(reward_name, beta)] = {
                name: value.detach().cpu().clone()
                for name, value in model.cnn.state_dict().items()
            }
            training_diagnostics.append(
                {
                    "reward": reward_name,
                    "beta_kl": beta,
                    "terminal_particle_reward": history[
                        "mean_reward"
                    ][-1],
                    "terminal_weighted_particle_reward": history[
                        "mean_reward_weighted"
                    ][-1],
                    "minimum_ess": min(history["ess"]),
                    "terminal_ess": history["ess"][-1],
                    "terminal_gamma": history["gamma"][-1],
                    "runtime_seconds": (
                        time.perf_counter() - trial_started
                    ),
                }
            )
            del tuner, energy, reference, model
            gc.collect()

    baseline_model = load_model().to("cpu")
    baseline_model.eval()
    baseline_samples = []
    for seed in EVALUATION_SEEDS:
        _seed_all(seed)
        baseline_samples.append(
            sample_original(
                baseline_model.cnn,
                n_samples=EVAL_SAMPLES_PER_SEED,
                steps=EVAL_STEPS,
                step_size=10.0,
                device=torch.device("cpu"),
            ).detach()
        )
    baseline_all = torch.cat(baseline_samples)
    baseline_digit = _digit_diagnostics(recognizer, baseline_all)

    raw_rows = []
    for reward_name in REWARD_NAMES:
        reward_fn = rewards[reward_name]
        baseline_rewards = torch.cat(
            [reward_fn(samples).detach() for samples in baseline_samples]
        )
        hacked = _hack_samples(reward_name, baseline_all.shape[0])
        hacked_digit = _digit_diagnostics(recognizer, hacked)
        hacked_digit["reward_mean"] = float(
            reward_fn(hacked).mean().item()
        )
        for beta in BETAS:
            model = load_model().to("cpu")
            model.cnn.load_state_dict(
                tuned_states[(reward_name, beta)]
            )
            model.eval()
            tuned_samples = []
            per_seed = []
            for seed, baseline_seed_samples in zip(
                EVALUATION_SEEDS, baseline_samples
            ):
                _seed_all(seed)
                samples = sample_original(
                    model.cnn,
                    n_samples=EVAL_SAMPLES_PER_SEED,
                    steps=EVAL_STEPS,
                    step_size=10.0,
                    device=torch.device("cpu"),
                ).detach()
                tuned_samples.append(samples)
                baseline_values = reward_fn(
                    baseline_seed_samples
                ).detach()
                tuned_values = reward_fn(samples).detach()
                per_seed.append(
                    {
                        "seed": seed,
                        "baseline_mean": float(
                            baseline_values.mean().item()
                        ),
                        "post_mean": float(tuned_values.mean().item()),
                        "paired_difference_mean": float(
                            (tuned_values - baseline_values)
                            .mean()
                            .item()
                        ),
                    }
                )
            tuned_all = torch.cat(tuned_samples)
            tuned_rewards = reward_fn(tuned_all).detach()
            raw_rows.append(
                {
                    "reward": reward_name,
                    "beta_kl": beta,
                    "baseline_reward": _reward_summary(
                        baseline_rewards
                    ),
                    "post_reward": _reward_summary(tuned_rewards),
                    "paired_reward_difference": _paired_interval(
                        baseline_rewards, tuned_rewards
                    ),
                    "evaluation_seeds": per_seed,
                    "digit_structure": {
                        "baseline": baseline_digit,
                        "tuned": _digit_diagnostics(
                            recognizer, tuned_all
                        ),
                        "reward_maximizing_control": hacked_digit,
                    },
                }
            )
            del model
            gc.collect()

    public_recognizer = {
        key: value
        for key, value in recognizer.items()
        if key
        not in {
            "model",
            "reference_features",
            "feature_mean",
            "feature_std",
        }
    }
    result = {
        "claim": (
            "Section 5.3 MNIST robustness under a mismatched "
            "pretraining/tuning kernel without reward hacking"
        ),
        "configuration": {
            "rewards": list(REWARD_NAMES),
            "reward_formulas": {
                "bright": "mean(clamp(x,-1,1))",
                "dark": "-mean(clamp(x,-1,1))",
                "lower_half": (
                    "0.5*(mean(bottom)-mean(top)) after clamping"
                ),
            },
            "beta_kl": list(BETAS),
            "training_seed": TRAIN_SEED,
            "outer_noise_seed": OUTER_NOISE_SEED,
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "n_particles": N_PARTICLES,
            "n_outer": N_OUTER,
            "tutorial_initialization_steps": (
                TUTORIAL_INITIALIZATION_STEPS
            ),
            "terminal_eval_samples_per_seed": (
                EVAL_SAMPLES_PER_SEED
            ),
            "terminal_eval_steps": EVAL_STEPS,
            "device": "cpu",
        },
        "source": {
            "official_notebook_sha256": sha256_file(NOTEBOOK),
            "pretrained_checkpoint_sha256": sha256_file(CHECKPOINT),
            "upstream_commit": (
                "62e4f8f07ae2705073388f5d2c4babf5c87b00be"
            ),
        },
        "kernel_mismatch_audit": {
            "pretraining_sampler": {
                "jitter_noise_std": 0.005,
                "gradient_clip": 0.03,
                "gradient_step_scale": 10.0,
                "state_clamp": [-1.0, 1.0],
                "transition_density": "not Gaussian",
            },
            "tuning_sampler": {
                "kernel": "pure Gaussian ULA",
                "step_size": TUNING_GAMMA,
                "noise_scale": 1.0,
                "gradient_clip": None,
                "state_clamp": None,
                "jitter_noise": None,
                "steps_per_outer": 1,
            },
        },
        "initial_particle_cache": initial_cache,
        "recognizer": public_recognizer,
        "training_diagnostics": training_diagnostics,
        "raw_rows": raw_rows,
        "runtime_seconds": time.perf_counter() - started,
    }
    checker = evaluate(result)
    result["independent_checker"] = checker
    result["negative_controls"] = {
        "label_swap": checker["negative_control"],
        "reward_maximizing_non_digit_controls": {
            row["reward"]: row["digit_structure"][
                "reward_maximizing_control"
            ]
            for row in raw_rows
        },
    }
    result["verdict"] = checker["verdict"]
    result["passed"] = checker["passed"]
    return result
