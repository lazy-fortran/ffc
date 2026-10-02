#!/usr/bin/env python3
"""Lock four full classified conformance reports to shared compiler inputs.

Report bytes and suite execution epochs remain distinct. Report paths are excluded
from the lock so relocated reports can be checked against the same artifact.
No compiler or corpus execution is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from conformance_observation import (
    ObservationError,
    SUITES,
    load_classification,
    selection_sha256,
    unique_object,
    write_atomic_bytes,
)


SHARED_FIELDS = (
    "ffc_revision",
    "ffc_source_sha256",
    "ffc_binary_sha256",
    "fortfront_revision",
    "fortfront_tree",
    "liric_revision",
    "liric_tree",
    "target_triple",
    "environment_sha256",
    "runtime_abi_sha256",
    "harness_sha256",
    "toolchain_sha256",
    "compiler_flags_sha256",
    "coverage_mode",
    "worktree",
)
SUITE_FIELDS = (
    "epoch_sha256",
    "corpus_revision",
    "corpus_tree",
    "corpus_files_sha256",
    "classification_mode",
    "classification_manifest_sha256",
    "observation_sha256",
)


def report_paths(arguments: list[str]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for argument in arguments:
        suite, separator, path = argument.partition("=")
        if not separator or not path:
            raise ObservationError("--report must be SUITE=PATH")
        if suite not in SUITES:
            raise ObservationError(f"unknown report suite: {suite}")
        if suite in reports:
            raise ObservationError(f"duplicate report suite: {suite}")
        reports[suite] = Path(path).resolve()
    missing = SUITES - reports.keys()
    if missing:
        raise ObservationError("missing report suites: " + ", ".join(sorted(missing)))
    return reports


def build_lock(reports: dict[str, Path]) -> dict[str, Any]:
    shared: dict[str, Any] | None = None
    locked_reports: list[dict[str, Any]] = []
    for suite in sorted(reports):
        data, cases, summary = load_classification(reports[suite], suite)
        if not cases:
            raise ObservationError(f"{suite}: report has no corpus cases")
        if not summary["full_run"]:
            raise ObservationError(f"{suite}: report is not a full run")
        if summary.get("sampled", False):
            raise ObservationError(f"{suite}: sampled report cannot lock a full epoch")
        if not summary["provenance_verified"]:
            raise ObservationError(f"{suite}: report provenance is not verified")
        current = {key: summary[key] for key in SHARED_FIELDS}
        if shared is None:
            shared = current
        else:
            for key in SHARED_FIELDS:
                if current[key] != shared[key]:
                    raise ObservationError(f"{suite}: shared input differs: {key}")
        locked_reports.append(
            {
                "suite": suite,
                "report_sha256": hashlib.sha256(data).hexdigest(),
                "selection_sha256": selection_sha256([case["file"] for case in cases]),
                **{key: summary[key] for key in SUITE_FIELDS},
            }
        )
    assert shared is not None
    shared["target_sha256"] = hashlib.sha256(
        shared["target_triple"].encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "report_schema_version": 2,
        "report_kind": "classification",
        "shared_inputs": shared,
        "reports": locked_reports,
    }


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def check_lock(path: Path, expected: dict[str, Any]) -> None:
    try:
        locked = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
    except (OSError, ValueError, UnicodeError) as error:
        raise ObservationError(f"{path}: cannot read provenance lock: {error}") from error
    if canonical_json(locked) != canonical_json(expected):
        raise ObservationError(f"{path}: provenance lock differs from current reports")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", required=True, metavar="SUITE=PATH",
        help="one schema-2 full classification report for each of the four suites",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="write a deterministic JSON lock")
    action.add_argument("--check", type=Path, help="verify a lock against current reports")
    args = parser.parse_args()
    try:
        reports = report_paths(args.report)
        if args.output is not None and args.output.resolve() in reports.values():
            raise ObservationError("lock output must not overwrite a report")
        locked = build_lock(reports)
        if args.check is not None:
            check_lock(args.check, locked)
        else:
            write_atomic_bytes(args.output, canonical_json(locked))
    except (ObservationError, OSError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
