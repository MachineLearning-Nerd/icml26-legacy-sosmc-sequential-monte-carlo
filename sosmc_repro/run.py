from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from sosmc_repro.ebm2d import run_2d_suite
from sosmc_repro.io import ARTIFACTS, ROOT, provenance, write_json, write_text
from sosmc_repro.langevin import run_wallclock
from sosmc_repro.theory import verify_equation_19, verify_proposition_1


def copy_contract(claim: int, artifact_dir: Path) -> None:
    contract_dir = ROOT / "evidence" / f"claim_{claim}"
    for source in contract_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, artifact_dir / source.name)


def main() -> int:
    started = time.perf_counter()
    config = json.loads((ROOT / "config" / "experiment.json").read_text())
    seed = int(config["seed"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    results: dict[int, dict] = {
        2: verify_proposition_1(),
        3: verify_equation_19(seed),
    }
    if config["stage"] in {"claim_4_wallclock", "claim_5_2d"}:
        results[4] = run_wallclock()
    if config["stage"] == "claim_5_2d":
        results[5] = run_2d_suite()
        results[1] = results[5]["algorithm1_result"]
    for claim, result in results.items():
        claim_dir = ARTIFACTS / f"claim_{claim}"
        claim_dir.mkdir(parents=True, exist_ok=True)
        copy_contract(claim, claim_dir)
        write_json(claim_dir / "raw_output.json", result)
        write_json(
            claim_dir / "independent_checker_output.json",
            {
                "checker": {
                    1: "official EBM trace plus independent weighted-gradient reconstruction",
                    2: "exact rational grid plus symbolic identities",
                    3: "symbolic Gaussian integration plus independent Monte Carlo",
                    4: "independent paired-seed wall-clock comparison",
                    5: "independent objective, tracking-error, and large-beta checks",
                }[claim],
                "passed": result["passed"],
                "details": result,
            },
        )
        write_json(
            claim_dir / "negative_control_output.json",
            result.get("negative_controls", result.get("independent_checker", {}).get("negative_control", {})),
        )
        write_json(claim_dir / "runtime.json", provenance(started, seed))
        write_text(
            claim_dir / "EVAL.md",
            f"# Claim {claim} evaluation\n\n"
            f"Verdict: **{result['verdict']}**\n\n"
            f"Passed: `{str(result['passed']).lower()}`\n",
        )

    if 1 not in results:
        claim_1_dir = ARTIFACTS / "claim_1"
        claim_1_dir.mkdir(parents=True, exist_ok=True)
        copy_contract(1, claim_1_dir)
        claim_1_result = {
            "verdict": "BLOCKED",
            "reason": (
                "The exact algorithm contract is frozen; a faithful EBM "
                "execution trace is required."
            ),
            "passed": False,
        }
        write_json(claim_1_dir / "raw_output.json", claim_1_result)
        write_text(
            claim_1_dir / "EVAL.md",
            "# Claim 1 evaluation\n\n"
            f"Verdict: **{claim_1_result['verdict']}**\n\n"
            f"{claim_1_result['reason']}\n",
        )

    summary = {
        "stage": config["stage"],
        "results": {str(k): v["verdict"] for k, v in results.items()},
        "all_accepted_passed": all(results[c]["passed"] for c in config["accepted_claims"]),
        "provenance": provenance(started, seed),
    }
    write_json(ARTIFACTS / "summary.json", summary)
    write_text(
        ARTIFACTS / "EVAL.md",
        "# SOSMC cumulative evaluation\n\n"
        f"Stage: `{config['stage']}`\n\n"
        + "\n".join(f"- Claim {k}: **{v['verdict']}**" for k, v in results.items())
        + f"\n\nAccepted checks passed: `{str(summary['all_accepted_passed']).lower()}`\n",
    )
    print("BEGIN_FULL_MACHINE_READABLE_RESULTS")
    print(json.dumps({"summary": summary, "claim_results": results}, indent=2, sort_keys=True))
    print("END_FULL_MACHINE_READABLE_RESULTS")
    return 0 if summary["all_accepted_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
