#!/usr/bin/env python3
"""Behavioral oracle for a portable four-suite classification provenance lock.

All reports below are synthetic fixtures, never production conformance data.
Execution epochs use the shard oracle's independent shell contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import conformance_shard_merge_oracle as fixture


PROJECT = Path(__file__).resolve().parent.parent
TOOL = PROJECT / "scripts" / "lock_conformance_epoch.py"
SUITE_FILES = {
    "fortfront-f90": ["a.f90", "b.f90"],
    "fortfront-lf": ["a.f90", "c.f90"],
    "lfortran": ["b.f90"],
    "gfortran-dg": ["c.f90", "d.f90"],
}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def make_report(
    root: Path, suite: str, changes: dict | None = None,
    files: list[str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    files = SUITE_FILES[suite] if files is None else files
    index = list(SUITE_FILES).index(suite)
    metadata = {
        "suite": suite,
        "full_run": True,
        "corpus_revision": str(6 + index) * 40,
        "corpus_tree": "abcd"[index] * 40,
        "skip_manifest_sha256": fixture.digest("skip:" + suite),
        "noref_manifest_sha256": fixture.digest("noref:" + suite),
    }
    metadata.update(changes or {})
    records = fixture.observation_records(
        files, fixture.selection_digest(files), metadata
    )
    for case in records[:-1]:
        case["suite"] = suite
        case["coverage_mode"] = records[-1]["coverage_mode"]
    observation = root / (suite + ".observation.jsonl")
    report = root / (suite + ".classification.jsonl")
    lookup = root / (suite + ".xfail.txt")
    lookup.write_text("b.f90\nd.f90\n", encoding="utf-8")
    fixture.write_jsonl(observation, records)
    classified = run([
        sys.executable, str(fixture.TOOL), "classify", "--suite", suite,
        str(observation), "--output", str(report),
        "--mode", "manifest", "--lookup", str(lookup),
        "--manifest-sha", fixture.digest("xfail:" + suite),
    ])
    assert classified.returncode == 0, classified.stderr
    return report


def command(reports: dict[str, Path], action: str, path: Path) -> list[str]:
    return [
        sys.executable, str(TOOL), action, str(path),
        *(part for suite, report in reports.items()
          for part in ("--report", suite + "=" + str(report))),
    ]


def reject(reports: dict[str, Path], output: Path, error: str) -> None:
    sentinel = b"previous lock survives\n"
    output.write_bytes(sentinel)
    result = run(command(reports, "--output", output))
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert error in result.stderr, result.stderr
    assert output.read_bytes() == sentinel, "rejection changed existing output"


def changed_report(report: Path, root: Path, edit) -> Path:
    records = [json.loads(line) for line in report.read_text().splitlines()]
    edit(records)
    output = root / "changed.jsonl"
    fixture.write_jsonl(output, records)
    return output


def check_corrupt_locks(reports: dict[str, Path], path: Path, locked: dict) -> None:
    for invalid in (True, 1.0, 2):
        bad_lock = copy.deepcopy(locked)
        bad_lock["schema_version"] = invalid
        path.write_text(json.dumps(bad_lock), encoding="utf-8")
        before = path.read_bytes()
        assert run(command(reports, "--check", path)).returncode == 2
        assert path.read_bytes() == before
    for invalid in (
        '{"schema_version":1,"schema_version":1}',
        '{"bad":"\\ud800"}',
        '{"unfinished":',
    ):
        path.write_text(invalid, encoding="utf-8")
        before = path.read_bytes()
        assert run(command(reports, "--check", path)).returncode == 2
        assert path.read_bytes() == before
    alias = path.parent / "output-alias.json"
    report = reports["lfortran"]
    alias.symlink_to(report)
    before = report.read_bytes()
    assert run(command(reports, "--output", alias)).returncode == 2
    assert alias.is_symlink() and report.read_bytes() == before


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="ffc-epoch-lock-oracle-", dir="/var/tmp"
    ) as temporary:
        root = Path(temporary)
        reports = {suite: make_report(root / "base", suite) for suite in SUITE_FILES}
        lock_path = root / "lock.json"
        result = run(command(reports, "--output", lock_path))
        assert result.returncode == 0, result.stderr
        original = lock_path.read_bytes()
        locked = json.loads(original)
        shared = locked["shared_inputs"]
        assert shared["ffc_revision"] == "1" * 40
        assert shared["ffc_source_sha256"] == fixture.digest("ffc-source")
        assert shared["ffc_binary_sha256"] == fixture.digest("ffc-binary")
        assert shared["worktree"] == "/fixture/ffc"
        assert shared["target_sha256"] == fixture.digest("x86_64-linux-gnu")
        assert [entry["suite"] for entry in locked["reports"]] == sorted(SUITE_FILES)
        assert len({entry["epoch_sha256"] for entry in locked["reports"]}) == 4
        for entry in locked["reports"]:
            suite = entry["suite"]
            report = reports[suite]
            summary = json.loads(report.read_text().splitlines()[-1])
            assert entry["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
            assert entry["selection_sha256"] == fixture.selection_digest(SUITE_FILES[suite])
            assert entry["corpus_revision"] == summary["corpus_revision"]
            assert entry["corpus_tree"] == summary["corpus_tree"]
            assert entry["corpus_files_sha256"] == fixture.selection_digest(SUITE_FILES[suite])
            assert entry["classification_manifest_sha256"] == fixture.digest("xfail:" + suite)
            assert entry["observation_sha256"] == summary["observation_sha256"]
            assert entry["epoch_sha256"] == fixture.execution_epoch(summary, SUITE_FILES[suite], True)
        assert run(command(reports, "--check", lock_path)).returncode == 0
        reversed_reports = dict(reversed(list(reports.items())))
        assert run(command(reversed_reports, "--output", lock_path)).returncode == 0
        assert lock_path.read_bytes() == original, "argument order changed lock"
        relocated = {}
        for suite, report in reports.items():
            relocated[suite] = root / (suite + ".copied.jsonl")
            relocated[suite].write_bytes(report.read_bytes())
        assert run(command(relocated, "--check", lock_path)).returncode == 0

        output = root / "rejected.json"
        reject({suite: path for suite, path in reports.items() if suite != "lfortran"}, output, "missing report suites")
        reject({**reports, "unknown": reports["lfortran"]}, output, "unknown report suite")
        reject({**reports, "lfortran": root / "missing.jsonl"}, output, "cannot read classification")
        extra = command(reports, "--output", output) + ["--report", "lfortran=" + str(reports["lfortran"])]
        result = run(extra)
        assert result.returncode == 2 and "duplicate report suite" in result.stderr
        assert output.read_bytes() == b"previous lock survives\n"

        suite = "lfortran"
        empty = make_report(root / "empty", suite, files=[])
        reject({**reports, suite: empty}, output, "report has no corpus cases")
        mismatches = {
            "ffc_revision": "f" * 40,
            "ffc_source_sha256": fixture.digest("other source"),
            "ffc_binary_sha256": fixture.digest("other binary"),
            "fortfront_revision": "e" * 40,
            "fortfront_tree": "f" * 40,
            "liric_revision": "e" * 40,
            "liric_tree": "f" * 40,
            "target_triple": "aarch64-linux-gnu",
            "environment_sha256": fixture.digest("other environment"),
            "runtime_abi_sha256": fixture.digest("other runtime"),
            "harness_sha256": fixture.digest("other harness"),
            "toolchain_sha256": fixture.digest("other toolchain"),
            "compiler_flags_sha256": fixture.digest("other global flags"),
            "coverage_mode": "llvm-profraw",
            "worktree": "/fixture/other-ffc",
        }
        for field, value in mismatches.items():
            replacement = make_report(root / field, suite, {field: value})
            reject({**reports, suite: replacement}, output, "shared input differs: " + field)
        for name, changes, error in (
            ("partial", {"full_run": False}, "not a full run"),
            ("unverified", {"provenance_verified": False}, "not verified"),
            ("sampled", {"sampled": True, "sample_size": 1, "sample_population": 2,
                         "sample_seed": 5, "sample_margin_pct": "10"}, "sampled report"),
        ):
            replacement = make_report(root / name, suite, changes)
            reject({**reports, suite: replacement}, output, error)

        def forge_epoch(records):
            for record in records:
                record["epoch_sha256"] = "0" * 64

        for name, edit, error in (
            ("epoch", forge_epoch, "execution epoch does not reconstruct"),
            ("summary", lambda records: records[-1].update({"pass": 100}), "SUMMARY pass mismatch"),
            ("kind", lambda records: records[-1].update({"report_kind": "observation"}), "not a classification"),
        ):
            altered = changed_report(reports[suite], root, edit)
            reject({**reports, suite: altered}, output, error)
        changed_bytes = root / "changed-bytes.jsonl"
        changed_bytes.write_bytes(reports[suite].read_bytes().replace(b"fixture observation", b"new fixture note"))
        result = run(command({**reports, suite: changed_bytes}, "--check", lock_path))
        assert result.returncode == 2 and "differs from current reports" in result.stderr
        assert lock_path.read_bytes() == original

        check_corrupt_locks(reports, lock_path, locked)
        sentinel = reports[suite].read_bytes()
        assert run(command(reports, "--output", reports[suite])).returncode == 2
        assert reports[suite].read_bytes() == sentinel
    print("PASS: four-suite lock accepts one shared provenance and rejects mixed or stale evidence")


if __name__ == "__main__":
    main()
