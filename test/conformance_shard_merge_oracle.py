#!/usr/bin/env python3
"""Independent behavioral oracle for disjoint conformance shard merging."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
TOOL = PROJECT_DIR / "scripts" / "conformance_observation.py"
SUITE = "fortfront-f90"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
EXPECTED_FILES = ["a.f90", "b.f90", "c.f90", "d.f90"]
CACHE_HIT_FILES = {"a.f90", "d.f90"}
STATUSES = {
    "a.f90": "PASS",
    "b.f90": "FAIL",
    "c.f90": "PASS",
    "d.f90": "FAIL",
    "e.f90": "PASS",
}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def case_flags_digest(ffc_flags: str, ref_flags: str) -> str:
    payload = f"ffc:{ffc_flags}\nref:{ref_flags}\n".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selection_data(file_names: list[str]) -> bytes:
    return "".join(f"{file_name}\n" for file_name in file_names).encode("utf-8")


def selection_digest(file_names: list[str]) -> str:
    return hashlib.sha256(selection_data(file_names)).hexdigest()


def execution_epoch(
    summary: dict[str, Any], file_names: list[str], full_run: bool
) -> str:
    """Compute the gauntlet contract through an independent shell oracle."""
    cache = "enabled" if summary["reference_cache_enabled"] else "disabled"
    script = r"""
{
    printf 'epoch_schema:2\n'
    printf 'suite:%s\nselection:%s\ncorpus:%s:%s:%s\n' \
        "$1" "$2" "$3" "$4" "$5"
    printf 'ffc:%s:%s:%s\n' "$6" "$7" "$8"
    printf 'fortfront:%s:%s\nliric:%s:%s\n' "$9" "${10}" "${11}" "${12}"
    printf 'target:%s\nenvironment:%s\nruntime:%s\nharness:%s\ntoolchain:%s\n' \
        "${13}" "${14}" "${15}" "${16}" "${17}"
    printf 'flags:%s\ntimeout:%s\nskip:%s\nnoref:%s\n' \
        "${18}" "${19}" "${20}" "${21}"
    printf 'cache:%s\nfull_run:%s\nworktree:%s\n' \
        "${22}" "${23}" "${24}"
} | sha256sum | cut -d ' ' -f 1
"""
    values = [
        summary["suite"],
        selection_digest(file_names),
        summary["corpus_revision"],
        summary["corpus_tree"],
        summary["corpus_files_sha256"],
        summary["ffc_revision"],
        summary["ffc_source_sha256"],
        summary["ffc_binary_sha256"],
        summary["fortfront_revision"],
        summary["fortfront_tree"],
        summary["liric_revision"],
        summary["liric_tree"],
        summary["target_triple"],
        summary["environment_sha256"],
        summary["runtime_abi_sha256"],
        summary["harness_sha256"],
        summary["toolchain_sha256"],
        summary["compiler_flags_sha256"],
        str(summary["timeout_seconds"]),
        summary["skip_manifest_sha256"],
        summary["noref_manifest_sha256"],
        cache,
        str(full_run).lower(),
        summary["worktree"],
    ]
    result = subprocess.run(
        ["bash", "-c", script, "epoch-oracle", *values],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def base_summary(corpus_digest: str) -> dict[str, Any]:
    return {
        "suite": SUITE,
        "status": "SUMMARY",
        "pass": 0,
        "xfail": 0,
        "xpass": 0,
        "fail": 0,
        "noref": 0,
        "skip": 0,
        "warning_unchecked": 0,
        "total": 0,
        "schema_version": 2,
        "full_run": False,
        "provenance_verified": True,
        "ffc_revision": "1" * 40,
        "ffc_source_sha256": digest("ffc-source"),
        "ffc_binary_sha256": digest("ffc-binary"),
        "fortfront_revision": "2" * 40,
        "fortfront_tree": "3" * 40,
        "liric_revision": "4" * 40,
        "liric_tree": "5" * 40,
        "corpus_revision": "6" * 40,
        "corpus_tree": "7" * 40,
        "corpus_files_sha256": corpus_digest,
        "worktree": "/fixture/ffc",
        "report_kind": "observation",
        "observation_schema_version": 2,
        "reference_compiler": "GNU Fortran fixture",
        "reference_cache_enabled": True,
        "reference_cache_hits": 0,
        "timeout_seconds": 5,
        "skip_manifest_sha256": digest("skip-manifest"),
        "noref_manifest_sha256": digest("noref-manifest"),
        "target_triple": "x86_64-linux-gnu",
        "environment_sha256": digest("environment"),
        "runtime_abi_sha256": digest("runtime"),
        "harness_sha256": digest("harness"),
        "toolchain_sha256": digest("toolchain"),
        "compiler_flags_sha256": digest(
            "ffc:default;reference:-w -J @private-module-dir\n"
        ),
        "coverage_mode": "none",
        "epoch_sha256": "0" * 64,
    }


def case_record(file_name: str, summary: dict[str, Any], epoch: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "suite": SUITE,
        "file": file_name,
        "status": STATUSES[file_name],
        "ffc_exit": 0,
        "ref_exit": 0,
        "note": "fixture observation",
        "epoch_sha256": epoch,
        "action": "compile-run",
        "source_sha256": digest(f"source:{file_name}"),
        "dependency_closure_sha256": digest(f"closure:{file_name}"),
        "ffc_flags": "default",
        "ref_flags": "-w -J @private-module-dir",
        "compiler_flags_sha256": case_flags_digest(
            "default", "-w -J @private-module-dir"
        ),
        "environment_sha256": summary["environment_sha256"],
        "target_triple": summary["target_triple"],
        "runtime_abi_sha256": summary["runtime_abi_sha256"],
        "harness_sha256": summary["harness_sha256"],
        "toolchain_sha256": summary["toolchain_sha256"],
        "phase": "compare",
        "diagnostic_signature_sha256": EMPTY_SHA256,
        "crash_signature_sha256": EMPTY_SHA256,
        "ffc_output_sha256": digest(f"ffc-output:{file_name}"),
        "ref_output_sha256": digest(f"ref-output:{file_name}"),
        "elapsed_ms": 4,
        "ffc_compile_ms": 1,
        "ffc_run_ms": 1,
        "ref_compile_ms": 1,
        "ref_run_ms": 1,
        "peak_rss_kb": 100,
        "semantic_tags": "none",
        "coverage_mode": "none",
        "coverage_sha256": EMPTY_SHA256,
    }
    for prefix in ("ffc_compile", "ffc_run", "ref_compile", "ref_run"):
        is_cached_reference = (
            file_name in CACHE_HIT_FILES and prefix.startswith("ref_")
        )
        record[f"{prefix}_action"] = (
            "cache-hit" if is_cached_reference else "executed"
        )
        record[f"{prefix}_exit"] = 0
        record[f"{prefix}_termination"] = "exit"
        record[f"{prefix}_signal"] = 0
    return record


def observation_records(
    file_names: list[str],
    corpus_digest: str,
    summary_changes: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary = base_summary(corpus_digest)
    if summary_changes:
        summary.update(summary_changes)
    epoch = execution_epoch(summary, file_names, summary["full_run"])
    summary["epoch_sha256"] = epoch
    cases = [case_record(file_name, summary, epoch) for file_name in file_names]
    summary["pass"] = sum(case["status"] == "PASS" for case in cases)
    summary["fail"] = sum(case["status"] == "FAIL" for case in cases)
    summary["skip"] = sum(case["status"] == "SKIP" for case in cases)
    summary["total"] = len(cases)
    summary["reference_cache_hits"] = sum(
        case["ref_compile_action"] == "cache-hit" for case in cases
    )
    return [*cases, summary]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def shard_command(
    expected: Path,
    expected_sha256: str,
    output: Path,
    shards: list[Path],
) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "merge",
        "--mode",
        "shards",
        "--suite",
        SUITE,
        "--expected-selection",
        str(expected),
        "--expected-selection-sha256",
        expected_sha256,
        "--output",
        str(output),
        *(str(shard) for shard in shards),
    ]


def require_rejection(command: list[str], output: Path, expected_error: str) -> None:
    sentinel = b"existing output remains\n"
    output.write_bytes(sentinel)
    result = run(command)
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert expected_error in result.stderr, result.stderr
    assert output.read_bytes() == sentinel


def main() -> None:
    corpus_digest = selection_digest(EXPECTED_FILES)
    with tempfile.TemporaryDirectory(prefix="ffc_shard_merge_") as temporary:
        scratch = Path(temporary)
        expected = scratch / "expected.txt"
        expected.write_bytes(selection_data(EXPECTED_FILES))
        shard_one = scratch / "shard-one.jsonl"
        shard_two = scratch / "shard-two.jsonl"
        write_jsonl(shard_one, observation_records(["a.f90", "c.f90"], corpus_digest))
        write_jsonl(shard_two, observation_records(["b.f90", "d.f90"], corpus_digest))
        original_shards = (shard_one.read_bytes(), shard_two.read_bytes())

        merged = scratch / "merged.jsonl"
        result = run(
            shard_command(expected, corpus_digest, merged, [shard_one, shard_two])
        )
        assert result.returncode == 0, (result.stdout, result.stderr)
        records = [json.loads(line) for line in merged.read_text().splitlines()]
        cases, summary = records[:-1], records[-1]
        expected_epoch = execution_epoch(
            base_summary(corpus_digest), EXPECTED_FILES, True
        )
        assert [case["file"] for case in cases] == EXPECTED_FILES
        assert [case["status"] for case in cases] == [
            "PASS",
            "FAIL",
            "PASS",
            "FAIL",
        ]
        assert summary["full_run"] is True
        assert summary["provenance_verified"] is True
        assert summary["corpus_files_sha256"] == corpus_digest
        assert summary["epoch_sha256"] == expected_epoch
        assert all(case["epoch_sha256"] == expected_epoch for case in cases)
        assert summary["pass"] == 2 and summary["fail"] == 2
        assert summary["total"] == 4 and summary["reference_cache_hits"] == 2
        assert "attempt_count" not in summary and "sampled" not in summary
        assert (shard_one.read_bytes(), shard_two.read_bytes()) == original_shards
        validate = run(
            [sys.executable, str(TOOL), "validate", "--suite", SUITE, str(merged)]
        )
        assert validate.returncode == 0, validate.stderr

        repeat = scratch / "repeat.jsonl"
        repeat_result = run(
            [
                sys.executable,
                str(TOOL),
                "merge",
                "--suite",
                SUITE,
                "--output",
                str(repeat),
                str(shard_one),
                str(shard_one),
            ]
        )
        assert repeat_result.returncode == 0, repeat_result.stderr
        repeat_summary = json.loads(repeat.read_text().splitlines()[-1])
        assert repeat_summary["attempt_count"] == 2
        assert repeat_summary["full_run"] is False
        assert repeat_summary["reference_cache_hits"] == 1

        forged_repeat = scratch / "forged-repeat-cache-hits.jsonl"
        repeat_records = [json.loads(line) for line in repeat.read_text().splitlines()]
        repeat_records[-1]["reference_cache_hits"] = 999
        write_jsonl(forged_repeat, repeat_records)
        forged_repeat_result = run(
            [
                sys.executable,
                str(TOOL),
                "validate",
                "--suite",
                SUITE,
                str(forged_repeat),
            ]
        )
        assert forged_repeat_result.returncode == 2
        assert (
            "SUMMARY reference_cache_hits mismatch" in forged_repeat_result.stderr
        )

        duplicate = scratch / "duplicate.jsonl"
        write_jsonl(
            duplicate,
            observation_records(["b.f90", "c.f90", "d.f90"], corpus_digest),
        )
        output = scratch / "reject-duplicate.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, duplicate]),
            output,
            "duplicate shard case: c.f90",
        )

        missing = scratch / "missing.jsonl"
        write_jsonl(missing, observation_records(["b.f90"], corpus_digest))
        output = scratch / "reject-missing.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, missing]),
            output,
            "missing=d.f90",
        )

        extra = scratch / "extra.jsonl"
        write_jsonl(
            extra,
            observation_records(["b.f90", "d.f90", "e.f90"], corpus_digest),
        )
        output = scratch / "reject-extra.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, extra]),
            output,
            "extra=e.f90",
        )

        mixed_pin = scratch / "mixed-pin.jsonl"
        write_jsonl(
            mixed_pin,
            observation_records(
                ["b.f90", "d.f90"],
                corpus_digest,
                {"ffc_revision": "a" * 40},
            ),
        )
        output = scratch / "reject-pin.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, mixed_pin]),
            output,
            "provenance differs from shard 1: ffc_revision",
        )

        mixed_harness = scratch / "mixed-harness.jsonl"
        write_jsonl(
            mixed_harness,
            observation_records(
                ["b.f90", "d.f90"],
                corpus_digest,
                {"harness_sha256": digest("other-harness")},
            ),
        )
        output = scratch / "reject-harness.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, mixed_harness]),
            output,
            "provenance differs from shard 1: harness_sha256",
        )

        mixed_schema = scratch / "mixed-schema.jsonl"
        write_jsonl(
            mixed_schema,
            observation_records(
                ["b.f90", "d.f90"], corpus_digest, {"schema_version": 1}
            ),
        )
        output = scratch / "reject-schema.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, mixed_schema]),
            output,
            "unknown report schema",
        )

        unverified = scratch / "unverified.jsonl"
        write_jsonl(
            unverified,
            observation_records(
                ["b.f90", "d.f90"],
                corpus_digest,
                {"provenance_verified": False},
            ),
        )
        output = scratch / "reject-unverified.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, unverified]),
            output,
            "lacks verified provenance",
        )

        cache_disabled = scratch / "cache-disabled.jsonl"
        write_jsonl(
            cache_disabled,
            observation_records(
                ["b.f90", "d.f90"],
                corpus_digest,
                {"reference_cache_enabled": False},
            ),
        )
        output = scratch / "reject-cache-disabled.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, cache_disabled]),
            output,
            "cache-hit evidence while reference cache is disabled",
        )

        already_full = scratch / "already-full.jsonl"
        write_jsonl(
            already_full,
            observation_records(EXPECTED_FILES, corpus_digest, {"full_run": True}),
        )
        output = scratch / "reject-full.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, already_full]),
            output,
            "is already a full run",
        )

        output = scratch / "reject-digest.jsonl"
        require_rejection(
            shard_command(expected, "0" * 64, output, [shard_one, shard_two]),
            output,
            "expected selection SHA-256 does not match file",
        )

        wrong_expected = scratch / "wrong-expected.txt"
        wrong_files = [*EXPECTED_FILES, "e.f90"]
        wrong_expected.write_bytes(selection_data(wrong_files))
        output = scratch / "reject-identity.jsonl"
        require_rejection(
            shard_command(
                wrong_expected,
                selection_digest(wrong_files),
                output,
                [shard_one, shard_two],
            ),
            output,
            "differs from locked corpus identity",
        )

        forged_epoch = scratch / "forged-epoch.jsonl"
        forged_records = observation_records(["b.f90", "d.f90"], corpus_digest)
        for record in forged_records:
            record["epoch_sha256"] = expected_epoch
        write_jsonl(forged_epoch, forged_records)
        output = scratch / "reject-forged-epoch.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, forged_epoch]),
            output,
            "execution epoch does not reconstruct",
        )

        forged_counts = scratch / "forged-counts.jsonl"
        count_records = observation_records(["b.f90", "d.f90"], corpus_digest)
        count_records[-1]["pass"] = 2
        count_records[-1]["fail"] = 0
        write_jsonl(forged_counts, count_records)
        output = scratch / "reject-forged-counts.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, forged_counts]),
            output,
            "SUMMARY pass mismatch",
        )

        forged_cache_hits = scratch / "forged-cache-hits.jsonl"
        cache_records = observation_records(["b.f90", "d.f90"], corpus_digest)
        cache_records[-1]["reference_cache_hits"] = 999
        write_jsonl(forged_cache_hits, cache_records)
        output = scratch / "reject-forged-cache-hits.jsonl"
        require_rejection(
            shard_command(
                expected, corpus_digest, output, [shard_one, forged_cache_hits]
            ),
            output,
            "SUMMARY reference_cache_hits mismatch",
        )

        forged_flags = scratch / "forged-row-flags.jsonl"
        flag_records = observation_records(["b.f90", "d.f90"], corpus_digest)
        flag_records[0]["compiler_flags_sha256"] = "0" * 64
        write_jsonl(forged_flags, flag_records)
        output = scratch / "reject-forged-row-flags.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, forged_flags]),
            output,
            "compiler_flags_sha256 does not match row flags",
        )

        nul_expected = scratch / "nul-expected.txt"
        nul_files = ["a.f90", "bad\0name.f90", "c.f90", "d.f90"]
        nul_expected.write_bytes(selection_data(nul_files))
        output = scratch / "reject-nul-selection.jsonl"
        require_rejection(
            shard_command(
                nul_expected,
                selection_digest(nul_files),
                output,
                [shard_one, shard_two],
            ),
            output,
            "invalid expected selection entry",
        )

        nul_case = scratch / "nul-case.jsonl"
        nul_case_records = observation_records(["b.f90", "d.f90"], corpus_digest)
        nul_case_records[0]["file"] = "bad\0name.f90"
        write_jsonl(nul_case, nul_case_records)
        output = scratch / "reject-nul-case.jsonl"
        require_rejection(
            shard_command(expected, corpus_digest, output, [shard_one, nul_case]),
            output,
            "invalid file field",
        )

        forged_aggregate = scratch / "forged-aggregate.jsonl"
        forged_records = copy.deepcopy(records)
        for record in forged_records:
            record["epoch_sha256"] = "f" * 64
        write_jsonl(forged_aggregate, forged_records)
        forged_validate = run(
            [
                sys.executable,
                str(TOOL),
                "validate",
                "--suite",
                SUITE,
                str(forged_aggregate),
            ]
        )
        assert forged_validate.returncode == 2
        assert "execution epoch does not reconstruct" in forged_validate.stderr

        forged_membership = scratch / "forged-membership.jsonl"
        membership_cases = copy.deepcopy(cases[:-1])
        membership_summary = copy.deepcopy(summary)
        membership_summary["fail"] = 1
        membership_summary["total"] = 3
        membership_summary["reference_cache_hits"] = 1
        membership_epoch = execution_epoch(
            membership_summary,
            [case["file"] for case in membership_cases],
            True,
        )
        membership_summary["epoch_sha256"] = membership_epoch
        for case in membership_cases:
            case["epoch_sha256"] = membership_epoch
        write_jsonl(forged_membership, [*membership_cases, membership_summary])
        membership_validate = run(
            [
                sys.executable,
                str(TOOL),
                "validate",
                "--suite",
                SUITE,
                str(forged_membership),
            ]
        )
        assert membership_validate.returncode == 2
        assert "full-run selection differs" in membership_validate.stderr

    print("PASS: shard merge requires exact membership and reconstructed provenance")


if __name__ == "__main__":
    main()
