from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".openresearch" / "artifacts"


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def cpu_allocation() -> dict[str, Any]:
    allocation: dict[str, Any] = {
        "os_cpu_count": os.cpu_count(),
        "affinity_count": None,
        "cgroup_quota_cores": None,
    }
    if hasattr(os, "sched_getaffinity"):
        allocation["affinity_count"] = len(os.sched_getaffinity(0))
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        quota, period = cpu_max.read_text(encoding="utf-8").split()
        if quota != "max":
            allocation["cgroup_quota_cores"] = int(quota) / int(period)
    return allocation


def provenance(started: float, seed: int) -> dict[str, Any]:
    return {
        "git_sha": git_sha(),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_allocation": cpu_allocation(),
        "runtime_seconds": time.perf_counter() - started,
        "cuda_visible": False,
        "device_policy": "CPU only",
    }
