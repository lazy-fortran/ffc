#!/usr/bin/env bash
# Create an expectation view from an existing immutable gauntlet observation.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib_expected_manifest.sh"
source "$SCRIPT_DIR/lib_conformance_observation.sh"

SUITES="fortfront-f90 fortfront-lf lfortran gfortran-dg"
SUITE=""
OBSERVATIONS=""
REPORT=""
XFAIL_MANIFEST=""
MODE="manifest"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --suite)
            [ $# -ge 2 ] || fail "--suite requires a name"
            SUITE="$2"; shift 2 ;;
        --observations)
            [ $# -ge 2 ] || fail "--observations requires a path"
            OBSERVATIONS="$2"; shift 2 ;;
        --report)
            [ $# -ge 2 ] || fail "--report requires a path"
            REPORT="$2"; shift 2 ;;
        --xfail-manifest)
            [ $# -ge 2 ] || fail "--xfail-manifest requires a path"
            XFAIL_MANIFEST="$2"; shift 2 ;;
        --no-xfail)
            MODE="xfail-disabled"; shift ;;
        *) fail "unknown option $1" ;;
    esac
done

[ -n "$SUITE" ] || fail "--suite is required. Choose from: $SUITES"
case "$SUITE" in
    fortfront-f90|fortfront-lf|lfortran|gfortran-dg) ;;
    *) fail "unknown suite '$SUITE'. Choose from: $SUITES" ;;
esac
[ -n "$OBSERVATIONS" ] || fail "--observations is required"
[ -n "$REPORT" ] || fail "--report is required"

if [ -z "$XFAIL_MANIFEST" ]; then
    safe_suite=${SUITE//-/_}
    XFAIL_MANIFEST="${FFC_XFAIL_MANIFEST:-$PROJECT_DIR/test/conformance/xfail_${safe_suite}.txt}"
fi

conformance_observation_classify "$OBSERVATIONS" "$REPORT" "$SUITE" \
    "$XFAIL_MANIFEST" "$MODE"
