from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEGIN = "BEGIN_FULL_MACHINE_READABLE_RESULTS"
END = "END_FULL_MACHINE_READABLE_RESULTS"


def main() -> None:
    text = sys.stdin.read()
    if BEGIN not in text or END not in text:
        raise SystemExit("machine-readable result markers not found")
    payload = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    result = json.loads(payload)

    release_raw = ROOT / "release" / "raw"
    release_raw.mkdir(parents=True, exist_ok=True)
    (release_raw / "full_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    artifacts = ROOT / ".openresearch" / "artifacts"
    for claim_id, claim in result["claim_results"].items():
        claim_dir = artifacts / f"claim_{claim_id}"
        claim_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = ROOT / "evidence" / f"claim_{claim_id}"
        for source in sorted(evidence_dir.iterdir()):
            if source.is_file():
                shutil.copy2(source, claim_dir / source.name)
        (claim_dir / "raw_output.json").write_text(
            json.dumps(claim, indent=2, sort_keys=True) + "\n"
        )
        (claim_dir / "independent_checker_output.json").write_text(
            json.dumps(
                claim.get("independent_checker", claim),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (claim_dir / "negative_control_output.json").write_text(
            json.dumps(
                claim.get(
                    "negative_controls",
                    claim.get("independent_checker", {}).get(
                        "negative_control", {}
                    ),
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (claim_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "run_id": (
                        "199652d8-ec32-4192-a79f-d76f5ea9a46f"
                    ),
                    "claim_runtime_seconds": claim.get(
                        "runtime_seconds"
                    ),
                    "run_provenance": result["summary"]["provenance"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (claim_dir / "EVAL.md").write_text(
            f"# Claim {claim_id} evaluation\n\n"
            f"Verdict: **{claim['verdict']}**\n\n"
            f"Passed: `{str(claim['passed']).lower()}`\n"
        )

    rows = result["claim_results"]["6"]["raw_rows"]
    with (release_raw / "claim6_reward_intervals.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "reward",
                "beta_kl",
                "difference_mean",
                "ci95_low",
                "ci95_high",
            ]
        )
        for row in rows:
            interval = row["paired_reward_difference"]
            writer.writerow(
                [
                    row["reward"],
                    row["beta_kl"],
                    interval["mean"],
                    interval["ci95_low"],
                    interval["ci95_high"],
                ]
            )


if __name__ == "__main__":
    main()
