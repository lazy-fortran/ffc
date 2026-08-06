#!/usr/bin/env bash
# lib_conformance_observation.sh: immutable gauntlet observation handling.
# Source this file; do not execute it directly.

set -uo pipefail

CONFORMANCE_OBSERVATION_SCRIPT_DIR="$(cd \
    "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFORMANCE_OBSERVATION_TOOL="$CONFORMANCE_OBSERVATION_SCRIPT_DIR/conformance_observation.py"

# conformance_observation_validate <observation_jsonl> <suite>
#                                  [case_list] [identity]
conformance_observation_validate() {
    local observations="$1" suite="$2"
    local case_list="${3:-}" identity="${4:-}"
    local -a arguments

    arguments=(validate --suite "$suite")
    if [ -n "$case_list" ]; then
        arguments+=(--case-list "$case_list")
    fi
    if [ -n "$identity" ]; then
        arguments+=(--identity "$identity")
    fi
    python3 "$CONFORMANCE_OBSERVATION_TOOL" "${arguments[@]}" \
        "$observations"
}

# conformance_observation_publish <staging> <output> <suite>
# Validates the exact staged bytes and atomically replaces the destination only
# after they form a complete observation.
conformance_observation_publish() {
    local staging="$1" output="$2" suite="$3"
    python3 "$CONFORMANCE_OBSERVATION_TOOL" publish --suite "$suite" \
        --output "$output" "$staging"
}

# conformance_observation_classify <observations> <report> <suite>
#                                  <manifest> <mode>
#
# The expectation manifest is copied before it is parsed. The classifier then
# reads and validates the observation exactly once, hashes those same bytes,
# and atomically publishes a view. It performs no compiler, program, selection,
# or oracle work. The return status is nonzero for FAIL, XPASS, or FLAKY.
conformance_observation_classify() {
    local observations="$1" report="$2" suite="$3" manifest="$4" mode="$5"
    local work manifest_snapshot lookup manifest_sha manifest_error_text
    local classification_status=0

    case "$mode" in
        manifest|xfail-disabled) ;;
        *)
            printf 'ERROR: unknown observation classification mode: %s\n' \
                "$mode" >&2
            return 1 ;;
    esac

    work=$(mktemp -d "${TMPDIR:-/tmp}/ffc_observation_classify_XXXXXX") || \
        return 2
    manifest_snapshot="$work/manifest.txt"
    lookup="$work/xfail_lookup.txt"
    : > "$lookup"
    if [ "$mode" = "manifest" ]; then
        if [ -f "$manifest" ]; then
            cp "$manifest" "$manifest_snapshot" || {
                rm -rf "$work"
                return 2
            }
        else
            : > "$manifest_snapshot"
        fi
        manifest_error_text=""
        if ! manifest_error_text=$(validate_expected_manifest \
                "$manifest_snapshot" "$lookup" 2>&1); then
            manifest_error_text=${manifest_error_text//"$manifest_snapshot"/"$manifest"}
            printf '%s\n' "$manifest_error_text" >&2
            rm -rf "$work"
            return 2
        fi
    else
        : > "$manifest_snapshot"
    fi
    manifest_sha=$(sha256sum "$manifest_snapshot" | cut -d ' ' -f 1)

    python3 "$CONFORMANCE_OBSERVATION_TOOL" classify --suite "$suite" \
        --mode "$mode" --lookup "$lookup" --manifest-sha "$manifest_sha" \
        --output "$report" "$observations" || classification_status=$?
    rm -rf "$work"
    return "$classification_status"
}

# conformance_observation_classification_counts <classification> <suite>
# Prints strict summary counters as one tab-delimited record. Validation here
# prevents display or gate logic from parsing JSON with filename-sensitive
# regular expressions.
conformance_observation_classification_counts() {
    local classification="$1" suite="$2"
    python3 "$CONFORMANCE_OBSERVATION_TOOL" classification-counts \
        --suite "$suite" "$classification"
}

# conformance_observation_merge <output> <suite> <attempt>...
conformance_observation_merge() {
    local output="$1" suite="$2"
    shift 2
    python3 "$CONFORMANCE_OBSERVATION_TOOL" merge --suite "$suite" \
        --output "$output" "$@"
}
