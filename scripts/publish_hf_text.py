from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-parent", required=True)
    args = parser.parse_args()

    allowed = [
        line.strip()
        for line in args.allowlist.read_text().splitlines()
        if line.strip()
    ]
    expected_hashes = {}
    for line in args.manifest.read_text().splitlines():
        digest, path = line.split("  ", 1)
        expected_hashes[path] = digest
    if set(allowed) != set(expected_hashes):
        raise SystemExit("allowlist and manifest paths differ")

    operations = []
    for relative in allowed:
        source = args.source / relative
        if not source.is_file():
            raise SystemExit(f"missing allowed file: {relative}")
        if sha256(source) != expected_hashes[relative]:
            raise SystemExit(f"hash mismatch before upload: {relative}")
        operations.append(
            CommitOperationAdd(path_in_repo=relative, path_or_fileobj=source)
        )

    api = HfApi()
    current = api.space_info(args.repo).sha
    if current != args.expected_parent:
        raise SystemExit(
            f"Space moved: expected {args.expected_parent}, found {current}"
        )
    result = api.create_commit(
        repo_id=args.repo,
        repo_type="space",
        operations=operations,
        commit_message="Publish claim-by-claim SOSMC reproduction",
        commit_description=(
            "Claims 1-5 verified and Claim 6 falsified in cumulative CPU "
            "evidence; includes raw outputs, checkers, controls, exact "
            "contracts, limitations, and historical preservation."
        ),
        parent_commit=current,
    )
    print(result.oid)
    print(result.commit_url)


if __name__ == "__main__":
    main()
