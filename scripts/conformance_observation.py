#!/usr/bin/env python3
"""Validate, classify, and merge ffc conformance observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


SUITES = {"fortfront-f90", "fortfront-lf", "lfortran", "gfortran-dg"}
RAW_STATUSES = {"PASS", "FAIL", "SKIP", "FLAKY"}
CLASSIFIED_STATUSES = RAW_STATUSES | {"XFAIL", "XPASS"}
NOREF_CATEGORIES = {
    "undefined-runtime-value",
    "missing-external-definition",
    "compile-only",
}
NOREF_REASONS = NOREF_CATEGORIES | {
    "reference-rejected",
    "reference-runtime-failure",
}
HEX40 = re.compile(r"^[0-9A-Fa-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TARGET_TRIPLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
SEMANTIC_TAG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
PHASES = {
    "compile",
    "run",
    "compare",
    "skip",
    "directive",
    "reference",
    "complete",
    "flaky",
}
COVERAGE_MODES = {"none", "llvm-profraw"}

CASE_DIGEST_FIELDS = (
    "source_sha256",
    "dependency_closure_sha256",
    "compiler_flags_sha256",
    "environment_sha256",
    "runtime_abi_sha256",
    "harness_sha256",
    "toolchain_sha256",
    "diagnostic_signature_sha256",
    "crash_signature_sha256",
    "ffc_output_sha256",
    "ref_output_sha256",
    "coverage_sha256",
)
CASE_TEXT_FIELDS = ("ffc_flags", "ref_flags", "target_triple")
CASE_METRIC_FIELDS = (
    "elapsed_ms",
    "ffc_compile_ms",
    "ffc_run_ms",
    "ref_compile_ms",
    "ref_run_ms",
    "peak_rss_kb",
)
LOCKED_CASE_FIELDS = (
    "source_sha256",
    "dependency_closure_sha256",
    "ffc_flags",
    "ref_flags",
    "compiler_flags_sha256",
    "environment_sha256",
    "target_triple",
    "runtime_abi_sha256",
    "harness_sha256",
    "toolchain_sha256",
    "semantic_tags",
    "coverage_mode",
)
DYNAMIC_DIGEST_FIELDS = (
    "diagnostic_signature_sha256",
    "crash_signature_sha256",
    "ffc_output_sha256",
    "ref_output_sha256",
    "coverage_sha256",
)
SUMMARY_CASE_LOCK_FIELDS = (
    "target_triple",
    "environment_sha256",
    "runtime_abi_sha256",
    "harness_sha256",
    "toolchain_sha256",
    "coverage_mode",
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

CASE_FIELDS = {
    "suite",
    "file",
    "status",
    "ffc_exit",
    "ref_exit",
    "note",
    "warning_expectation",
    "noref",
    "noref_reason",
    "noref_manifest_category",
    "attempts",
    "observed",
    *CASE_DIGEST_FIELDS,
    *CASE_TEXT_FIELDS,
    *CASE_METRIC_FIELDS,
    "phase",
    "semantic_tags",
    "coverage_mode",
}
CLASSIFICATION_CASE_FIELDS = CASE_FIELDS | {"observed_status", "expectation"}
SUMMARY_FIELDS = {
    "suite",
    "status",
    "pass",
    "xfail",
    "xpass",
    "fail",
    "noref",
    "skip",
    "warning_unchecked",
    "total",
    "flaky",
    "schema_version",
    "full_run",
    "sampled",
    "sample_size",
    "sample_population",
    "sample_seed",
    "sample_margin_pct",
    "provenance_verified",
    "ffc_revision",
    "ffc_source_sha256",
    "ffc_binary_sha256",
    "fortfront_revision",
    "fortfront_tree",
    "liric_revision",
    "liric_tree",
    "corpus_revision",
    "corpus_tree",
    "corpus_files_sha256",
    "worktree",
    "report_kind",
    "observation_schema_version",
    "reference_compiler",
    "reference_cache_enabled",
    "reference_cache_hits",
    "timeout_seconds",
    "skip_manifest_sha256",
    "noref_manifest_sha256",
    "attempt_count",
    "target_triple",
    "environment_sha256",
    "runtime_abi_sha256",
    "harness_sha256",
    "toolchain_sha256",
    "compiler_flags_sha256",
    "coverage_mode",
}
CLASSIFICATION_SUMMARY_FIELDS = SUMMARY_FIELDS | {
    "classification_mode",
    "observation_sha256",
    "classification_manifest_sha256",
}
IDENTITY_FIELDS = (
    "suite",
    "schema_version",
    "full_run",
    "sampled",
    "sample_size",
    "sample_population",
    "sample_seed",
    "provenance_verified",
    "ffc_revision",
    "ffc_source_sha256",
    "ffc_binary_sha256",
    "fortfront_revision",
    "fortfront_tree",
    "liric_revision",
    "liric_tree",
    "corpus_revision",
    "corpus_tree",
    "corpus_files_sha256",
    "worktree",
    "report_kind",
    "observation_schema_version",
    "reference_compiler",
    "reference_cache_enabled",
    "timeout_seconds",
    "skip_manifest_sha256",
    "noref_manifest_sha256",
    "target_triple",
    "environment_sha256",
    "runtime_abi_sha256",
    "harness_sha256",
    "toolchain_sha256",
    "compiler_flags_sha256",
    "coverage_mode",
)


class ObservationError(Exception):
    """A malformed or incompatible observation."""


class DuplicateKeyError(ValueError):
    """A JSON object contains a duplicate key."""


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def require_exact_type(
    record: dict[str, Any], key: str, expected: type, location: str
) -> Any:
    if key not in record:
        raise ObservationError(f"{location}: missing field: {key}")
    value = record[key]
    if expected is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    else:
        valid = type(value) is expected
    if not valid:
        raise ObservationError(f"{location}: wrong field type: {key}")
    return value


def nonnegative_int(record: dict[str, Any], key: str, location: str) -> int:
    value = require_exact_type(record, key, int, location)
    if value < 0:
        raise ObservationError(f"{location}: negative counter: {key}")
    return value


def validate_digest(record: dict[str, Any], key: str, location: str) -> None:
    value = require_exact_type(record, key, str, location)
    if not HEX64.fullmatch(value):
        raise ObservationError(f"{location}: malformed digest: {key}")


def validate_revision(record: dict[str, Any], key: str, location: str) -> None:
    value = require_exact_type(record, key, str, location)
    if not HEX40.fullmatch(value):
        raise ObservationError(f"{location}: malformed revision: {key}")


def validate_nonempty_text(
    record: dict[str, Any], key: str, location: str
) -> str:
    value = require_exact_type(record, key, str, location)
    if not value or "\0" in value or "\n" in value or "\r" in value:
        raise ObservationError(f"{location}: invalid text field: {key}")
    return value


def validate_case_provenance(
    record: dict[str, Any], status: str, location: str
) -> None:
    for key in CASE_DIGEST_FIELDS:
        validate_digest(record, key, location)
    for key in CASE_TEXT_FIELDS:
        validate_nonempty_text(record, key, location)
    for key in CASE_METRIC_FIELDS:
        nonnegative_int(record, key, location)

    target_triple = record["target_triple"]
    if not TARGET_TRIPLE.fullmatch(target_triple):
        raise ObservationError(f"{location}: invalid target_triple")

    phase = validate_nonempty_text(record, "phase", location)
    if phase not in PHASES:
        raise ObservationError(f"{location}: invalid phase: {phase}")
    if (status == "FLAKY") != (phase == "flaky"):
        raise ObservationError(f"{location}: phase/status mismatch")

    semantic_tags = validate_nonempty_text(record, "semantic_tags", location)
    tags = semantic_tags.split(",")
    if (
        any(not SEMANTIC_TAG.fullmatch(tag) for tag in tags)
        or len(set(tags)) != len(tags)
    ):
        raise ObservationError(f"{location}: invalid semantic_tags")

    coverage_mode = validate_nonempty_text(record, "coverage_mode", location)
    if coverage_mode not in COVERAGE_MODES:
        raise ObservationError(
            f"{location}: invalid coverage_mode: {coverage_mode}"
        )
    if coverage_mode == "none" and record["coverage_sha256"] != EMPTY_SHA256:
        raise ObservationError(
            f"{location}: coverage_sha256 must hash empty content when coverage is none"
        )


def parse_jsonl(data: bytes, source: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ObservationError(f"{source}: invalid UTF-8: {error}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        location = f"{source}:{line_number}"
        if not line.strip():
            raise ObservationError(f"{location}: blank JSONL record")
        try:
            record = json.loads(line, object_pairs_hook=unique_object)
        except (json.JSONDecodeError, DuplicateKeyError) as error:
            raise ObservationError(f"{location}: invalid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ObservationError(f"{location}: record is not a JSON object")
        records.append(record)
    return records


def validate_case(
    record: dict[str, Any], suite: str, location: str
) -> tuple[str, str, int, int]:
    unknown = set(record) - CASE_FIELDS
    if unknown:
        raise ObservationError(
            f"{location}: unknown observation field: {sorted(unknown)[0]}"
        )
    if require_exact_type(record, "suite", str, location) != suite:
        raise ObservationError(f"{location}: mixed or unexpected suite")
    file_name = require_exact_type(record, "file", str, location)
    if not file_name or "\n" in file_name or "\r" in file_name:
        raise ObservationError(f"{location}: invalid file field")
    status = require_exact_type(record, "status", str, location)
    if status not in RAW_STATUSES:
        raise ObservationError(
            f"{location}: classified status in raw observation: {status}"
        )
    note = require_exact_type(record, "note", str, location)
    validate_case_provenance(record, status, location)

    has_ffc = "ffc_exit" in record
    has_ref = "ref_exit" in record
    if has_ffc != has_ref:
        raise ObservationError(f"{location}: result has only one exit field")
    if has_ffc:
        ffc_exit = require_exact_type(record, "ffc_exit", int, location)
        ref_exit = require_exact_type(record, "ref_exit", int, location)
        if ffc_exit < -1 or ref_exit < -1:
            raise ObservationError(f"{location}: invalid exit status")
    elif status not in {"SKIP", "FLAKY"} and not (
        status == "FAIL" and "directive" in note
    ):
        raise ObservationError(f"{location}: result is missing exit fields")

    if status == "FLAKY":
        if nonnegative_int(record, "attempts", location) < 2:
            raise ObservationError(f"{location}: FLAKY needs two attempts")
        observed = require_exact_type(record, "observed", str, location)
        if "|" not in observed:
            raise ObservationError(f"{location}: FLAKY needs differing states")
    elif "attempts" in record or "observed" in record:
        raise ObservationError(f"{location}: attempt fields outside FLAKY")

    noref = 0
    if "noref" in record:
        if require_exact_type(record, "noref", bool, location) is not True:
            raise ObservationError(f"{location}: noref must be true")
        if status != "PASS":
            raise ObservationError(f"{location}: raw NOREF must be PASS")
        reason = require_exact_type(record, "noref_reason", str, location)
        if reason not in NOREF_REASONS:
            raise ObservationError(f"{location}: invalid noref reason: {reason}")
        if reason == "reference-rejected" and record["ref_exit"] == 0:
            raise ObservationError(
                f"{location}: reference-rejected NOREF has ref_exit=0"
            )
        if reason == "reference-runtime-failure" and (
            record["ffc_exit"] != 0 or record["ref_exit"] == 0
        ):
            raise ObservationError(
                f"{location}: invalid reference-runtime-failure exits"
            )
        noref = 1
    elif "noref_reason" in record:
        raise ObservationError(f"{location}: noref_reason without noref")

    if "noref_manifest_category" in record:
        category = require_exact_type(
            record, "noref_manifest_category", str, location
        )
        if category not in NOREF_CATEGORIES:
            raise ObservationError(
                f"{location}: invalid noref manifest category: {category}"
            )

    warning = 0
    if "warning_expectation" in record:
        warning_value = require_exact_type(
            record, "warning_expectation", str, location
        )
        if warning_value != "unchecked" or suite != "gfortran-dg":
            raise ObservationError(f"{location}: invalid warning expectation")
        warning = 1
    return file_name, status, noref, warning


def validate_classification_case(
    record: dict[str, Any], suite: str, location: str
) -> tuple[str, str, int, int]:
    unknown = set(record) - CLASSIFICATION_CASE_FIELDS
    if unknown:
        raise ObservationError(
            f"{location}: unknown classification field: {sorted(unknown)[0]}"
        )
    observed = require_exact_type(record, "observed_status", str, location)
    expectation = require_exact_type(record, "expectation", str, location)
    if observed not in RAW_STATUSES:
        raise ObservationError(f"{location}: invalid observed status: {observed}")
    if expectation not in {"none", "xfail"}:
        raise ObservationError(f"{location}: invalid expectation: {expectation}")
    expected_status = observed
    if expectation == "xfail" and observed == "PASS":
        expected_status = "XPASS"
    elif expectation == "xfail" and observed == "FAIL":
        expected_status = "XFAIL"
    status = require_exact_type(record, "status", str, location)
    if status not in CLASSIFIED_STATUSES or status != expected_status:
        raise ObservationError(
            f"{location}: classified status does not match observation"
        )
    if expectation == "xfail" and "noref_manifest_category" in record:
        raise ObservationError(
            "files cannot be both xfail and noref: " + record.get("file", "")
        )

    raw = dict(record)
    raw.pop("observed_status")
    raw.pop("expectation")
    raw["status"] = observed
    file_name, _, noref, warning = validate_case(raw, suite, location)
    return file_name, status, noref, warning


def validate_summary_metadata(
    summary: dict[str, Any], location: str, case_count: int
) -> None:
    require_exact_type(summary, "full_run", bool, location)
    require_exact_type(summary, "provenance_verified", bool, location)
    for key in (
        "ffc_revision",
        "fortfront_revision",
        "fortfront_tree",
        "liric_revision",
        "liric_tree",
        "corpus_revision",
        "corpus_tree",
    ):
        validate_revision(summary, key, location)
    for key in (
        "ffc_source_sha256",
        "ffc_binary_sha256",
        "corpus_files_sha256",
        "skip_manifest_sha256",
        "noref_manifest_sha256",
        "environment_sha256",
        "runtime_abi_sha256",
        "harness_sha256",
        "toolchain_sha256",
        "compiler_flags_sha256",
    ):
        validate_digest(summary, key, location)
    for key in ("worktree", "reference_compiler"):
        if not require_exact_type(summary, key, str, location):
            raise ObservationError(f"{location}: empty field: {key}")
    require_exact_type(summary, "reference_cache_enabled", bool, location)
    nonnegative_int(summary, "reference_cache_hits", location)
    if nonnegative_int(summary, "timeout_seconds", location) < 1:
        raise ObservationError(f"{location}: timeout_seconds must be positive")

    target_triple = validate_nonempty_text(
        summary, "target_triple", location
    )
    if not TARGET_TRIPLE.fullmatch(target_triple):
        raise ObservationError(f"{location}: invalid target_triple")
    coverage_mode = validate_nonempty_text(summary, "coverage_mode", location)
    if coverage_mode not in COVERAGE_MODES:
        raise ObservationError(
            f"{location}: invalid coverage_mode: {coverage_mode}"
        )

    sampled = summary.get("sampled", False)
    if not isinstance(sampled, bool):
        raise ObservationError(f"{location}: wrong field type: sampled")
    sample_fields = {
        "sample_size",
        "sample_population",
        "sample_seed",
        "sample_margin_pct",
    }
    if sampled:
        if nonnegative_int(summary, "sample_size", location) != case_count:
            raise ObservationError(f"{location}: sample_size mismatch")
        nonnegative_int(summary, "sample_population", location)
        nonnegative_int(summary, "sample_seed", location)
        margin = require_exact_type(summary, "sample_margin_pct", str, location)
        try:
            numeric_margin = float(margin)
        except ValueError as error:
            raise ObservationError(
                f"{location}: invalid sample_margin_pct"
            ) from error
        if not math.isfinite(numeric_margin) or numeric_margin < 0:
            raise ObservationError(f"{location}: invalid sample_margin_pct")
    elif sample_fields & set(summary):
        raise ObservationError(f"{location}: sample fields without sampled=true")
    if "attempt_count" in summary and nonnegative_int(
        summary, "attempt_count", location
    ) < 2:
        raise ObservationError(f"{location}: attempt_count must be at least two")


def validate_summary_case_provenance(
    cases: list[dict[str, Any]], summary: dict[str, Any], location: str
) -> None:
    for index, case in enumerate(cases, 1):
        for key in SUMMARY_CASE_LOCK_FIELDS:
            if case[key] != summary[key]:
                raise ObservationError(
                    f"{location}: case {index} differs from SUMMARY: {key}"
                )


def validate_summary(
    summary: dict[str, Any],
    suite: str,
    location: str,
    counts: dict[str, int],
    noref_count: int,
    warning_count: int,
    case_count: int,
) -> None:
    unknown = set(summary) - SUMMARY_FIELDS
    if unknown:
        raise ObservationError(
            f"{location}: unknown observation SUMMARY field: {sorted(unknown)[0]}"
        )
    if require_exact_type(summary, "suite", str, location) != suite:
        raise ObservationError(f"{location}: mixed or unexpected SUMMARY suite")
    if require_exact_type(summary, "status", str, location) != "SUMMARY":
        raise ObservationError(f"{location}: invalid SUMMARY status")
    if nonnegative_int(summary, "schema_version", location) != 2:
        raise ObservationError(f"{location}: unknown report schema")
    if require_exact_type(summary, "report_kind", str, location) != "observation":
        raise ObservationError(f"{location}: SUMMARY is not an observation")
    if nonnegative_int(summary, "observation_schema_version", location) != 2:
        raise ObservationError(f"{location}: unknown observation schema")

    expected = {
        "pass": counts.get("PASS", 0),
        "xfail": 0,
        "xpass": 0,
        "fail": counts.get("FAIL", 0),
        "noref": noref_count,
        "skip": counts.get("SKIP", 0),
        "warning_unchecked": warning_count,
        "total": case_count,
    }
    for key, value in expected.items():
        if nonnegative_int(summary, key, location) != value:
            raise ObservationError(f"{location}: SUMMARY {key} mismatch")
    flaky_count = counts.get("FLAKY", 0)
    recorded_flaky = summary.get("flaky", 0)
    if not isinstance(recorded_flaky, int) or isinstance(recorded_flaky, bool):
        raise ObservationError(f"{location}: wrong field type: flaky")
    if recorded_flaky != flaky_count:
        raise ObservationError(f"{location}: SUMMARY flaky mismatch")
    validate_summary_metadata(summary, location, case_count)


def validate_classification_summary(
    summary: dict[str, Any],
    suite: str,
    location: str,
    counts: dict[str, int],
    noref_count: int,
    warning_count: int,
    case_count: int,
) -> None:
    unknown = set(summary) - CLASSIFICATION_SUMMARY_FIELDS
    if unknown:
        raise ObservationError(
            f"{location}: unknown classification SUMMARY field: "
            f"{sorted(unknown)[0]}"
        )
    if require_exact_type(summary, "suite", str, location) != suite:
        raise ObservationError(f"{location}: mixed or unexpected SUMMARY suite")
    if require_exact_type(summary, "status", str, location) != "SUMMARY":
        raise ObservationError(f"{location}: invalid SUMMARY status")
    if nonnegative_int(summary, "schema_version", location) != 2:
        raise ObservationError(f"{location}: unknown report schema")
    if require_exact_type(summary, "report_kind", str, location) != "classification":
        raise ObservationError(f"{location}: SUMMARY is not a classification")
    if nonnegative_int(summary, "observation_schema_version", location) != 2:
        raise ObservationError(f"{location}: unknown observation schema")

    expected = {
        "pass": counts.get("PASS", 0),
        "xfail": counts.get("XFAIL", 0),
        "xpass": counts.get("XPASS", 0),
        "fail": counts.get("FAIL", 0),
        "noref": noref_count,
        "skip": counts.get("SKIP", 0),
        "warning_unchecked": warning_count,
        "total": case_count,
    }
    for key, value in expected.items():
        if nonnegative_int(summary, key, location) != value:
            raise ObservationError(f"{location}: SUMMARY {key} mismatch")
    flaky_count = counts.get("FLAKY", 0)
    recorded_flaky = summary.get("flaky", 0)
    if not isinstance(recorded_flaky, int) or isinstance(recorded_flaky, bool):
        raise ObservationError(f"{location}: wrong field type: flaky")
    if recorded_flaky != flaky_count:
        raise ObservationError(f"{location}: SUMMARY flaky mismatch")

    mode = require_exact_type(summary, "classification_mode", str, location)
    if mode not in {"manifest", "xfail-disabled"}:
        raise ObservationError(f"{location}: invalid classification mode")
    validate_digest(summary, "observation_sha256", location)
    validate_digest(summary, "classification_manifest_sha256", location)
    validate_summary_metadata(summary, location, case_count)


def load_observation(path: Path, suite: str) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ObservationError(f"{path}: cannot read observation: {error}") from error
    records = parse_jsonl(data, str(path))
    if not records:
        raise ObservationError(f"{path}: missing SUMMARY")

    cases: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    seen: set[str] = set()
    counts: dict[str, int] = {}
    noref_count = 0
    warning_count = 0
    for index, record in enumerate(records, 1):
        location = f"{path}:{index}"
        if record.get("status") == "SUMMARY":
            if summary is not None:
                raise ObservationError(f"{location}: duplicate SUMMARY")
            if index != len(records):
                raise ObservationError(f"{location}: SUMMARY must be final")
            summary = record
            continue
        if summary is not None:
            raise ObservationError(f"{location}: record follows SUMMARY")
        file_name, status, is_noref, warning = validate_case(
            record, suite, location
        )
        if file_name in seen:
            raise ObservationError(f"{location}: duplicate case: {file_name}")
        seen.add(file_name)
        counts[status] = counts.get(status, 0) + 1
        noref_count += is_noref
        warning_count += warning
        cases.append(record)
    if summary is None:
        raise ObservationError(f"{path}: missing SUMMARY")
    validate_summary(
        summary,
        suite,
        f"{path}:{len(records)}",
        counts,
        noref_count,
        warning_count,
        len(cases),
    )
    validate_summary_case_provenance(cases, summary, str(path))
    flaky_cases = [case for case in cases if case["status"] == "FLAKY"]
    if flaky_cases and "attempt_count" not in summary:
        raise ObservationError(f"{path}: FLAKY observation lacks attempt_count")
    if "attempt_count" in summary:
        attempt_count = summary["attempt_count"]
        for case in flaky_cases:
            if case["attempts"] != attempt_count:
                raise ObservationError(
                    f"{path}: FLAKY attempts differ from SUMMARY attempt_count"
                )
    return data, cases, summary


def load_classification(
    path: Path, suite: str
) -> tuple[bytes, list[dict[str, Any]], dict[str, Any]]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ObservationError(
            f"{path}: cannot read classification: {error}"
        ) from error
    records = parse_jsonl(data, str(path))
    if not records:
        raise ObservationError(f"{path}: missing SUMMARY")

    cases: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    seen: set[str] = set()
    counts: dict[str, int] = {}
    noref_count = 0
    warning_count = 0
    for index, record in enumerate(records, 1):
        location = f"{path}:{index}"
        if record.get("status") == "SUMMARY":
            if summary is not None:
                raise ObservationError(f"{location}: duplicate SUMMARY")
            if index != len(records):
                raise ObservationError(f"{location}: SUMMARY must be final")
            summary = record
            continue
        file_name, status, is_noref, warning = validate_classification_case(
            record, suite, location
        )
        if file_name in seen:
            raise ObservationError(f"{location}: duplicate case: {file_name}")
        seen.add(file_name)
        counts[status] = counts.get(status, 0) + 1
        noref_count += is_noref
        warning_count += warning
        cases.append(record)
    if summary is None:
        raise ObservationError(f"{path}: missing SUMMARY")
    validate_classification_summary(
        summary,
        suite,
        f"{path}:{len(records)}",
        counts,
        noref_count,
        warning_count,
        len(cases),
    )
    validate_summary_case_provenance(cases, summary, str(path))
    flaky_cases = [case for case in cases if case["status"] == "FLAKY"]
    if flaky_cases and "attempt_count" not in summary:
        raise ObservationError(f"{path}: FLAKY classification lacks attempt_count")
    if "attempt_count" in summary:
        attempt_count = summary["attempt_count"]
        for case in flaky_cases:
            if case["attempts"] != attempt_count:
                raise ObservationError(
                    f"{path}: FLAKY attempts differ from SUMMARY attempt_count"
                )
    return data, cases, summary


def identity(summary: dict[str, Any]) -> str:
    values = {key: summary.get(key) for key in IDENTITY_FIELDS}
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def write_atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for record in records:
                json.dump(record, stream, ensure_ascii=False, separators=(",", ":"))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def status_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for case in cases:
        status = case["status"]
        result[status] = result.get(status, 0) + 1
    return result


def update_summary_counts(
    summary: dict[str, Any], cases: list[dict[str, Any]]
) -> None:
    counts = status_counts(cases)
    summary["pass"] = counts.get("PASS", 0)
    summary["xfail"] = counts.get("XFAIL", 0)
    summary["xpass"] = counts.get("XPASS", 0)
    summary["fail"] = counts.get("FAIL", 0)
    summary["noref"] = sum(case.get("noref") is True for case in cases)
    summary["skip"] = counts.get("SKIP", 0)
    summary["warning_unchecked"] = sum(
        "warning_expectation" in case for case in cases
    )
    summary["total"] = len(cases)
    if counts.get("FLAKY", 0):
        summary["flaky"] = counts["FLAKY"]
    else:
        summary.pop("flaky", None)
    if summary.get("sampled") is True:
        population = summary["sample_population"]
        total = len(cases)
        if total <= 0 or population <= 1 or total >= population:
            margin = 0.0
        else:
            rate = counts.get("PASS", 0) / total
            correction = (population - total) / (population - 1)
            margin = 1.96 * math.sqrt(
                rate * (1.0 - rate) / total * correction
            ) * 100.0
        summary["sample_margin_pct"] = f"{margin:.1f}"


def validate_in_memory_observation(
    cases: list[dict[str, Any]], summary: dict[str, Any], suite: str
) -> None:
    seen: set[str] = set()
    counts: dict[str, int] = {}
    noref_count = 0
    warning_count = 0
    for index, case in enumerate(cases, 1):
        file_name, status, noref, warning = validate_case(
            case, suite, f"merged observation:{index}"
        )
        if file_name in seen:
            raise ObservationError(f"merged observation:{index}: duplicate case")
        seen.add(file_name)
        counts[status] = counts.get(status, 0) + 1
        noref_count += noref
        warning_count += warning
    validate_summary(
        summary,
        suite,
        "merged observation:SUMMARY",
        counts,
        noref_count,
        warning_count,
        len(cases),
    )
    validate_summary_case_provenance(cases, summary, "merged observation")


def validate_in_memory_classification(
    cases: list[dict[str, Any]], summary: dict[str, Any], suite: str
) -> None:
    seen: set[str] = set()
    counts: dict[str, int] = {}
    noref_count = 0
    warning_count = 0
    for index, case in enumerate(cases, 1):
        file_name, status, noref, warning = validate_classification_case(
            case, suite, f"classification:{index}"
        )
        if file_name in seen:
            raise ObservationError(f"classification:{index}: duplicate case")
        seen.add(file_name)
        counts[status] = counts.get(status, 0) + 1
        noref_count += noref
        warning_count += warning
    validate_classification_summary(
        summary,
        suite,
        "classification:SUMMARY",
        counts,
        noref_count,
        warning_count,
        len(cases),
    )
    validate_summary_case_provenance(cases, summary, "classification")


def command_validate(args: argparse.Namespace) -> int:
    _, cases, summary = load_observation(args.observation, args.suite)
    if args.case_list:
        write_atomic_text(
            args.case_list, "".join(f"{case['file']}\n" for case in cases)
        )
    if args.identity:
        write_atomic_text(args.identity, identity(summary) + "\n")
    return 0


def command_publish(args: argparse.Namespace) -> int:
    input_path = args.observation.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ObservationError(
            f"published observation must use a distinct staging path: {input_path}"
        )
    data, _, _ = load_observation(input_path, args.suite)
    write_atomic_bytes(output_path, data)
    return 0


def command_classify(args: argparse.Namespace) -> int:
    input_path = args.observation.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ObservationError(
            f"classified report must not overwrite observations: {input_path}"
        )
    if not HEX64.fullmatch(args.manifest_sha):
        raise ObservationError("classification manifest SHA-256 is malformed")
    data, cases, original_summary = load_observation(input_path, args.suite)
    expected: set[str] = set()
    if args.mode == "manifest":
        try:
            expected = set(args.lookup.read_text(encoding="utf-8").splitlines())
        except OSError as error:
            raise ObservationError(f"cannot read expectation lookup: {error}") from error
        expected.discard("")

    classified_cases: list[dict[str, Any]] = []
    for raw_case in cases:
        case = dict(raw_case)
        observed = case["status"]
        expectation = "xfail" if case["file"] in expected else "none"
        if expectation == "xfail" and "noref_manifest_category" in case:
            raise ObservationError(
                "files cannot be both xfail and noref: " + case["file"]
            )
        if expectation == "xfail" and observed == "PASS":
            case["status"] = "XPASS"
        elif expectation == "xfail" and observed == "FAIL":
            case["status"] = "XFAIL"
        case["observed_status"] = observed
        case["expectation"] = expectation
        classified_cases.append(case)

    summary = dict(original_summary)
    update_summary_counts(summary, classified_cases)
    summary["report_kind"] = "classification"
    summary["classification_mode"] = args.mode
    summary["observation_sha256"] = hashlib.sha256(data).hexdigest()
    summary["classification_manifest_sha256"] = args.manifest_sha
    validate_in_memory_classification(classified_cases, summary, args.suite)
    write_atomic_jsonl(output_path, [*classified_cases, summary])
    return int(
        any(
            case["status"] in {"FAIL", "XPASS", "FLAKY"}
            for case in classified_cases
        )
    )


def command_classification_counts(args: argparse.Namespace) -> int:
    _, _, summary = load_classification(args.classification, args.suite)
    values = (
        summary["pass"],
        summary["xfail"],
        summary["xpass"],
        summary["fail"],
        summary.get("flaky", 0),
        summary["noref"],
        summary["skip"],
        summary["warning_unchecked"],
        summary["total"],
        summary.get("sample_population", 0),
    )
    print("\t".join(str(value) for value in values))
    return 0


def canonical_behavior(case: dict[str, Any]) -> str:
    """Return the repeat oracle, excluding non-behavioral resource metrics."""
    behavior = {
        key: value
        for key, value in case.items()
        if key not in CASE_METRIC_FIELDS
    }
    return json.dumps(behavior, sort_keys=True, separators=(",", ":"))


def aggregate_attempt_digests(
    field: str, attempt_cases: list[dict[str, Any]]
) -> str:
    """Bind an aggregate digest to its field, attempt order, and every value."""
    payload = json.dumps(
        {
            "field": field,
            "attempts": [case[field] for case in attempt_cases],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def command_merge(args: argparse.Namespace) -> int:
    if len(args.observations) < 2:
        raise ObservationError("repeat merge requires at least two observations")
    loaded = [load_observation(path.resolve(), args.suite) for path in args.observations]
    for attempt, (_, _, summary) in enumerate(loaded, 1):
        if "attempt_count" in summary:
            raise ObservationError(
                f"attempt {attempt} is already a merged observation"
            )
    first_cases = loaded[0][1]
    first_files = [case["file"] for case in first_cases]
    first_identity = identity(loaded[0][2])
    for attempt, (_, cases, summary) in enumerate(loaded[1:], 2):
        files = [case["file"] for case in cases]
        if files != first_files:
            raise ObservationError(
                f"attempt {attempt} case set/order differs from attempt 1"
            )
        if identity(summary) != first_identity:
            raise ObservationError(
                f"attempt {attempt} metadata differs from attempt 1"
            )

    merged_cases: list[dict[str, Any]] = []
    for case_index, file_name in enumerate(first_files):
        attempt_cases = [entry[1][case_index] for entry in loaded]
        for key in (
            *LOCKED_CASE_FIELDS,
            "warning_expectation",
            "noref_manifest_category",
        ):
            values = [case.get(key) for case in attempt_cases]
            if any(value != values[0] for value in values[1:]):
                raise ObservationError(
                    f"case metadata differs across attempts: {file_name}: {key}"
                )
        canonical_cases = [canonical_behavior(case) for case in attempt_cases]
        states: list[str] = []
        for case in attempt_cases:
            state = case["status"]
            if state not in states:
                states.append(state)
        if all(value == canonical_cases[0] for value in canonical_cases[1:]):
            merged_cases.append(dict(attempt_cases[0]))
        else:
            if len(states) == 1:
                states = [
                    f"{case['status']}@"
                    + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
                    for case, canonical in zip(attempt_cases, canonical_cases)
                ]
                states = list(dict.fromkeys(states))
            flaky_case: dict[str, Any] = {
                "suite": args.suite,
                "file": file_name,
                "status": "FLAKY",
                "note": f"unstable across {len(loaded)} attempts",
                "attempts": len(loaded),
                "observed": "|".join(states),
                "phase": "flaky",
            }
            for key in LOCKED_CASE_FIELDS:
                flaky_case[key] = attempt_cases[0][key]
            for key in DYNAMIC_DIGEST_FIELDS:
                if key == "coverage_sha256" and (
                    flaky_case["coverage_mode"] == "none"
                ):
                    flaky_case[key] = EMPTY_SHA256
                else:
                    flaky_case[key] = aggregate_attempt_digests(
                        key, attempt_cases
                    )
            for key in CASE_METRIC_FIELDS[:-1]:
                flaky_case[key] = sum(case[key] for case in attempt_cases)
            flaky_case["peak_rss_kb"] = max(
                case["peak_rss_kb"] for case in attempt_cases
            )
            for key in ("warning_expectation", "noref_manifest_category"):
                if key in attempt_cases[0]:
                    flaky_case[key] = attempt_cases[0][key]
            merged_cases.append(flaky_case)

    summary = dict(loaded[0][2])
    update_summary_counts(summary, merged_cases)
    summary["attempt_count"] = len(loaded)
    summary["reference_cache_hits"] = sum(
        entry[2]["reference_cache_hits"] for entry in loaded
    )
    output_path = args.output.resolve()
    if output_path in {path.resolve() for path in args.observations}:
        raise ObservationError("merged output must not overwrite an attempt")
    validate_in_memory_observation(merged_cases, summary, args.suite)
    write_atomic_jsonl(output_path, [*merged_cases, summary])
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--suite", choices=sorted(SUITES), required=True)
    validate.add_argument("--case-list", type=Path)
    validate.add_argument("--identity", type=Path)
    validate.add_argument("observation", type=Path)
    validate.set_defaults(function=command_validate)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--suite", choices=sorted(SUITES), required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("observation", type=Path)
    publish.set_defaults(function=command_publish)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--suite", choices=sorted(SUITES), required=True)
    classify.add_argument(
        "--mode", choices=("manifest", "xfail-disabled"), required=True
    )
    classify.add_argument("--lookup", type=Path, required=True)
    classify.add_argument("--manifest-sha", required=True)
    classify.add_argument("--output", type=Path, required=True)
    classify.add_argument("observation", type=Path)
    classify.set_defaults(function=command_classify)

    classification_counts = subparsers.add_parser("classification-counts")
    classification_counts.add_argument(
        "--suite", choices=sorted(SUITES), required=True
    )
    classification_counts.add_argument("classification", type=Path)
    classification_counts.set_defaults(function=command_classification_counts)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--suite", choices=sorted(SUITES), required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("observations", nargs="+", type=Path)
    merge.set_defaults(function=command_merge)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.function(args)
    except ObservationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"ERROR: observation I/O failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
