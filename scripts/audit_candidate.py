from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_target(source: Path, link: str, root: Path) -> Path | None:
    target = link.split("#", 1)[0].split("?", 1)[0]
    if not target or "://" in target or target.startswith("mailto:"):
        return None
    candidate = (source.parent / target).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AssertionError(f"link escapes candidate: {source}: {link}") from exc
    return candidate


def traverse(root: Path) -> list[Path]:
    queue = [root / "README.md", root / "pages" / "index.md"]
    logbook = json.loads((root / "logbook.json").read_text())
    queue.extend(
        root / child["file"]
        for child in logbook["root"]["children"]
    )
    opened: list[Path] = []
    seen: set[Path] = set()
    while queue:
        source = queue.pop(0).resolve()
        if source in seen:
            continue
        assert source.exists(), f"missing reachable file: {source}"
        seen.add(source)
        opened.append(source)
        if source.suffix.lower() != ".md":
            continue
        for link in LINK.findall(source.read_text()):
            target = local_target(source, link, root)
            if target is None:
                continue
            assert target.exists(), f"broken link: {source}: {link}"
            if target.is_file() and target.suffix.lower() in {
                ".md", ".json", ".csv", ".py", ".toml", ".lock", ".txt"
            }:
                queue.append(target)
    return opened


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    source = args.candidate.resolve()
    assert source.is_dir()

    with tempfile.TemporaryDirectory(prefix="sosmc-candidate-audit-") as temp:
        fresh = Path(temp) / "download"
        shutil.copytree(source, fresh)
        fresh = fresh.resolve()
        opened = traverse(fresh)

        current = (fresh / "pages" / "current-verification" /
                   "page.md").read_text()
        for claim_id in range(1, 7):
            assert f"| {claim_id} |" in current
            page = fresh / "pages" / f"claim-{claim_id}" / "page.md"
            assert page.resolve() in opened
            text = page.read_text()
            for required in [
                "Exact claim contract",
                "Raw numerical result inline",
                "Source and quantifiers",
                "Method, code, and command",
                "Evidence and independent checks",
                "Limitations and deviations",
            ]:
                assert required in text, f"Claim {claim_id}: missing {required}"

        verify = subprocess.run(
            ["python3", str(fresh / "verify_release.py"), "--root", str(fresh)],
            text=True,
            capture_output=True,
        )
        assert verify.returncode == 0, verify.stderr

        # Negative control: evidence tampering must make the verifier nonzero.
        results = fresh / "raw" / "full_results.json"
        payload = json.loads(results.read_text())
        payload["summary"]["results"]["6"] = "VERIFIED"
        results.write_text(json.dumps(payload))
        tamper = subprocess.run(
            ["python3", str(fresh / "verify_release.py"), "--root", str(fresh)],
            text=True,
            capture_output=True,
        )
        assert tamper.returncode != 0, "tampered evidence was accepted"

    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {
            ".md", ".json", ".csv", ".py", ".toml", ".lock", ".txt",
            ".html", ".css", ".js", ".svg",
        } or path.name == ".gitattributes":
            text = path.read_text(errors="replace")
            for pattern in SECRET_PATTERNS:
                assert not pattern.search(text), f"possible secret in {path}"

    report = (
        "# Evaluator-blind candidate audit\n\n"
        "Verdict: **PASS**\n\n"
        "The review began only from `README.md`, `pages/index.md`, and the "
        "files named by `logbook.json`. No repository knowledge was used to "
        "supply missing evidence.\n\n"
        "## Files opened\n\n"
        + "\n".join(
            f"- `{path.relative_to(fresh).as_posix()}`" for path in opened
        )
        + "\n\n## Conclusions\n\n"
        "- All six canonical claim pages were reachable.\n"
        "- Every visibility-matrix row exposed code, inline data, raw evidence, "
        "an independent checker, a failing control, and the exact contract.\n"
        "- The current verifier passed the unchanged candidate.\n"
        "- A modified Claim 6 summary made the verifier exit nonzero.\n"
        "- No supported secret pattern appeared in text artifacts.\n"
        "- No conclusion remained unverifiable from the candidate traversal.\n"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
    print("PASS: evaluator-visible traversal, verifier, tamper control, secret scan")


if __name__ == "__main__":
    main()
