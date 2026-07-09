#!/usr/bin/env bash
# conformance_check.sh: single-command conformance gate.
#
# Does a clean build, runs every available suite, fails on any FAIL or XPASS,
# and prints the promotable XPASS list.
#
# Usage:
#   scripts/conformance_check.sh [--no-build] [--suite SUITE] [SELECTION]
#
# Options:
#   --no-build   skip build step (use existing ffc binary)
#   --suite S    run only one suite instead of all available suites
#   --file PATH  forward one suite-relative file (repeatable)
#   --files-from PATH  forward a named-file list (repeatable)
#
# This script is the documented routine contributors run before pushing
# and after dependency (fortfront, liric) updates.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_conformance.sh"

PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMARY_REPO_ROOT="$(resolve_primary_checkout_root "$PROJECT_DIR")"
CORPUS_PARENT="$(dirname "$PRIMARY_REPO_ROOT")"
GAUNTLET="$SCRIPT_DIR/conformance_gauntlet.sh"

# Defaults
NO_BUILD=0
SINGLE_SUITE=""
TIMEOUT=5
NAMED_ARGS=()

# Argument parsing
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build)
            NO_BUILD=1; shift ;;
        --suite)
            SINGLE_SUITE="$2"; shift 2 ;;
        --file)
            if [ $# -lt 2 ]; then
                echo "ERROR: --file requires a path" >&2; exit 1
            fi
            NAMED_ARGS+=("--file" "$2"); shift 2 ;;
        --files-from)
            if [ $# -lt 2 ]; then
                echo "ERROR: --files-from requires a path" >&2; exit 1
            fi
            list_path="$2"
            case "$list_path" in /*) ;; *) list_path="$PWD/$list_path" ;; esac
            NAMED_ARGS+=("--files-from" "$list_path"); shift 2 ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        *)
            echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done

if [ "${#NAMED_ARGS[@]}" -gt 0 ] && [ -z "$SINGLE_SUITE" ]; then
    echo "ERROR: --file and --files-from require --suite" >&2
    exit 1
fi

# Determine suites to run
ALL_SUITES="fortfront-f90 fortfront-lf lfortran gfortran-dg"

if [ -n "$SINGLE_SUITE" ]; then
    SUITES="$SINGLE_SUITE"
else
    # Only include a suite if its root directory exists.
    SUITES=""
    for s in $ALL_SUITES; do
        case "$s" in
            fortfront-f90) root="${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/f90" ;;
            fortfront-lf)  root="${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/lf" ;;
            lfortran)      root="${FFC_LFORTRAN_DIR:-$CORPUS_PARENT/lfortran}/integration_tests" ;;
            gfortran-dg)   root="${FFC_GFORTRAN_DG_DIR:-$CORPUS_PARENT/gcc/gcc/testsuite/gfortran.dg}" ;;
        esac
        if [ -d "$root" ]; then
            SUITES="$SUITES $s"
        else
            echo "SKIP: suite $s not found at $root (run scripts/fetch_corpora.sh or set env var)"
        fi
    done
fi

SUITES=$(echo "$SUITES" | xargs)  # trim whitespace

if [ -z "$SUITES" ]; then
    echo "ERROR: no suites available. Set FFC_FORTFRONT_DIR or run scripts/fetch_corpora.sh" >&2
    exit 1
fi

# Build step
if [ "$NO_BUILD" -eq 0 ]; then
    echo "=== Building ffc ==="
    cd "$PROJECT_DIR"
    if command -v fo >/dev/null 2>&1; then
        fo build
    else
        fpm build --profile release
    fi
    echo ""
fi

# Resolve ffc binary
FFC_BIN=$(find_ffc) || {
    echo "ERROR: ffc binary not found after build" >&2
    exit 1
}
echo "Using ffc: $FFC_BIN"
echo ""

# Run each suite
HAS_FAIL=0
HAS_XPASS=0
XPASS_FILES=""

for SUITE in $SUITES; do
    REPORT="/tmp/ffc_conformance_${SUITE}.jsonl"
    LOG="/tmp/ffc_conformance_${SUITE}.out"

    echo "=== Running suite: $SUITE ==="

    rm -f "$REPORT" "$LOG"

    gauntlet_args=(--suite "$SUITE" --ffc "$FFC_BIN" \
        --report "$REPORT" --timeout "$TIMEOUT")
    gauntlet_args+=("${NAMED_ARGS[@]}")
    if bash "$GAUNTLET" "${gauntlet_args[@]}" > "$LOG" 2>&1; then
        suite_exit=0
    else
        suite_exit=$?
    fi

    # Print the log (summary line and any FAIL/XPASS)
    grep -E '(===|PASS=|FAIL:|XPASS:|ERROR:)' "$LOG" || true
    echo ""

    # Parse summary
    if [ -f "$REPORT" ]; then
        summary=$(grep '"status":"SUMMARY"' "$REPORT" || echo "")
        if [ -n "$summary" ]; then
            fail_count=$(echo "$summary" | grep -o '"fail":[0-9]*' | grep -o '[0-9]*')
            xpass_count=$(echo "$summary" | grep -o '"xpass":[0-9]*' | grep -o '[0-9]*')
            pass_count=$(echo "$summary" | grep -o '"pass":[0-9]*' | grep -o '[0-9]*')
            xfail_count=$(echo "$summary" | grep -o '"xfail":[0-9]*' | grep -o '[0-9]*')
            total_count=$(echo "$summary" | grep -o '"total":[0-9]*' | grep -o '[0-9]*')

            echo "  $SUITE: PASS=$pass_count XFAIL=$xfail_count XPASS=$xpass_count FAIL=$fail_count TOTAL=$total_count"

            if [ "${fail_count:-0}" -gt 0 ]; then
                HAS_FAIL=1
            fi
            if [ "${xpass_count:-0}" -gt 0 ]; then
                HAS_XPASS=1
                # Collect XPASS file names
                xpass_list=$(grep '"status":"XPASS"' "$REPORT" | \
                    grep -o '"file":"[^"]*"' | \
                    sed 's/"file":"//;s/"//' || true)
                if [ -n "$xpass_list" ]; then
                    XPASS_FILES="$XPASS_FILES
$SUITE:
$xpass_list"
                fi
            fi
        fi
    fi

    if [ "$suite_exit" -ne 0 ]; then
        HAS_FAIL=1
    fi
done

echo ""
echo "=== Conformance check summary ==="

# Print XPASS list if any
if [ -n "$XPASS_FILES" ]; then
    echo ""
    echo "Promotable XPASS entries (remove from xfail manifest to promote):"
    echo "$XPASS_FILES"
    echo ""
fi

# Exit code
if [ "$HAS_FAIL" -ne 0 ]; then
    echo "FAIL: one or more suites have FAIL records"
    exit 1
fi

if [ "$HAS_XPASS" -ne 0 ]; then
    echo "FAIL: one or more suites have XPASS records (manifest drift — promote or investigate)"
    exit 1
fi

echo "PASS: all suites clean (no FAIL, no XPASS)"
exit 0
