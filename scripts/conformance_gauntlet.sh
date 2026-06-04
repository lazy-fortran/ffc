#!/usr/bin/env bash
# conformance_gauntlet.sh: drive external Fortran corpora through ffc.
#
# Usage:
#   scripts/conformance_gauntlet.sh --suite SUITE [OPTIONS]
#
# Suites:
#   fortfront-f90   FortFront standard-mode examples (../fortfront/examples/f90)
#   fortfront-lf    FortFront lazy-mode examples (../fortfront/examples/lf)
#   lfortran        LFortran integration tests
#   gfortran-dg     GCC gfortran.dg testsuite
#
# Options:
#   --suite SUITE       required
#   --ffc PATH          path to ffc binary (auto-discovered if omitted)
#   --report PATH       JSONL report path (default: /tmp/ffc_gauntlet_<suite>.jsonl)
#   --max-files N       only test the first N files (for smoke runs)
#   --timeout N         per-file timeout in seconds (default: 5)
#
# Environment variables (suite roots):
#   FFC_FORTFRONT_DIR   default: ../fortfront
#   FFC_LFORTRAN_DIR    default: ../lfortran
#   FFC_GFORTRAN_DG_DIR default: ../gcc/gcc/testsuite/gfortran.dg
#
# No foreign source files are copied into this repository.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_conformance.sh"

# Defaults
SUITES="fortfront-f90 fortfront-lf lfortran gfortran-dg"
FFC_BIN=""
REPORT=""
MAX_FILES=""
TIMEOUT=5
HAS_FAIL=0

# Argument parsing
SUITE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --suite)
            SUITE="$2"; shift 2 ;;
        --ffc)
            FFC_BIN="$2"; shift 2 ;;
        --report)
            REPORT="$2"; shift 2 ;;
        --max-files)
            MAX_FILES="$2"; shift 2 ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        *)
            echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done

if [ -z "$SUITE" ]; then
    echo "ERROR: --suite is required. Choose from: $SUITES" >&2
    exit 1
fi

# Validate suite name
case "$SUITE" in
    fortfront-f90|fortfront-lf|lfortran|gfortran-dg) ;;
    *) echo "ERROR: unknown suite '$SUITE'. Choose from: $SUITES" >&2; exit 1 ;;
esac

# Resolve report path
if [ -z "$REPORT" ]; then
    REPORT="/tmp/ffc_gauntlet_${SUITE}.jsonl"
fi

# Ensure report directory exists
mkdir -p "$(dirname "$REPORT")"

# Resolve suite root
resolve_suite_root() {
    case "$SUITE" in
        fortfront-f90)
            echo "${FFC_FORTFRONT_DIR:-../fortfront}/examples/f90" ;;
        fortfront-lf)
            echo "${FFC_FORTFRONT_DIR:-../fortfront}/examples/lf" ;;
        lfortran)
            echo "${FFC_LFORTRAN_DIR:-../lfortran}/integration_tests" ;;
        gfortran-dg)
            echo "${FFC_GFORTRAN_DG_DIR:-../gcc/gcc/testsuite/gfortran.dg}" ;;
    esac
}

# Resolve xfail manifest
resolve_xfail_manifest() {
    local safe_suite
    safe_suite=${SUITE//-/_}
    echo "test/conformance/xfail_${safe_suite}.txt"
}

# File extension for the suite
file_extension() {
    case "$SUITE" in
        fortfront-f90) echo "f90" ;;
        fortfront-lf)  echo "lf" ;;
        lfortran)      echo "f90" ;;
        gfortran-dg)   echo "f90" ;;
    esac
}

# Lazy suites have no gfortran reference.
is_lazy_suite() {
    [ "$SUITE" = "fortfront-lf" ]
}

# Resolve ffc
if [ -z "$FFC_BIN" ]; then
    FFC_BIN=$(find_ffc) || exit 1
fi

# Setup
SUITE_ROOT=$(resolve_suite_root)
XFAIL_MANIFEST=$(resolve_xfail_manifest)
EXT=$(file_extension)
TMPDIR_WORK=$(mktemp -d /tmp/ffc_gauntlet_XXXXXX)
trap 'rm -rf "$TMPDIR_WORK"' EXIT

# Counters
PASS_COUNT=0
XFAIL_COUNT=0
XPASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
TOTAL_COUNT=0

# Clear report
> "$REPORT"

# Check suite root exists.
if [ ! -d "$SUITE_ROOT" ]; then
    echo "SKIP: $SUITE not found at $SUITE_ROOT"
    printf '{"suite":"%s","status":"SUMMARY","pass":%d,"xfail":%d,"xpass":%d,"fail":%d}\n' \
        "$SUITE" "$PASS_COUNT" "$XFAIL_COUNT" "$XPASS_COUNT" "$FAIL_COUNT" >> "$REPORT"
    exit 0
fi

# Collect files.
FILE_LIST="$TMPDIR_WORK/files.txt"
find "$SUITE_ROOT" -maxdepth 1 -name "*.$EXT" -type f | sort > "$FILE_LIST"

if [ "$MAX_FILES" -gt 0 ] 2>/dev/null; then
    head -n "$MAX_FILES" "$FILE_LIST" > "$TMPDIR_WORK/files_limited.txt"
    mv "$TMPDIR_WORK/files_limited.txt" "$FILE_LIST"
fi

FILE_COUNT=$(wc -l < "$FILE_LIST")
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "SKIP: no .${EXT} files found in $SUITE_ROOT"
    printf '{"suite":"%s","status":"SUMMARY","pass":%d,"xfail":%d,"xpass":%d,"fail":%d}\n' \
        "$SUITE" "$PASS_COUNT" "$XFAIL_COUNT" "$XPASS_COUNT" "$FAIL_COUNT" >> "$REPORT"
    exit 0
fi

echo "Running $SUITE: $FILE_COUNT files, timeout=${TIMEOUT}s, ffc=$FFC_BIN"

# Process each file.
while IFS= read -r full_path; do
    [ -z "$full_path" ] && continue
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    basename_file=$(basename "$full_path")
    # Suite-relative path is the basename for single-depth search
    rel_path="$basename_file"

    ffc_exe="$TMPDIR_WORK/ffc_${TOTAL_COUNT}"
    ref_exe="$TMPDIR_WORK/ref_${TOTAL_COUNT}"
    ffc_out="$TMPDIR_WORK/ffc_out_${TOTAL_COUNT}"
    ref_out="$TMPDIR_WORK/ref_out_${TOTAL_COUNT}"

    rm -f "$ffc_exe" "$ref_exe" "$ffc_out" "$ref_out"

    ffc_exit=-1
    ref_exit=-1
    status=""
    note=""

    # Step 1: compile with ffc
    if compile_with_ffc "$full_path" "$ffc_exe" "$FFC_BIN"; then
        ffc_exit=0
    else
        ffc_exit=1
    fi

    # Step 2: if ffc failed, classify immediately
    if [ "$ffc_exit" -ne 0 ]; then
        if check_xfail "$XFAIL_MANIFEST" "$rel_path"; then
            status="XFAIL"
            note="listed in xfail manifest"
            XFAIL_COUNT=$((XFAIL_COUNT + 1))
            printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
                "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
            continue
        else
            status="FAIL"
            note="ffc compilation failed"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
                "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
            echo "  FAIL: $rel_path (ffc failed)"
            continue
        fi
    fi

    # Step 3: run ffc binary
    run_capture "$ffc_exe" "$ffc_out" "$TIMEOUT"
    ffc_exit=$?

    if [ "$ffc_exit" -ne 0 ]; then
        if check_xfail "$XFAIL_MANIFEST" "$rel_path"; then
            status="XFAIL"
            note="listed in xfail manifest (runtime failure)"
            XFAIL_COUNT=$((XFAIL_COUNT + 1))
            printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
                "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
            continue
        else
            status="FAIL"
            note="ffc runtime failed (exit $ffc_exit)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
                "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
            echo "  FAIL: $rel_path (runtime exit $ffc_exit)"
            continue
        fi
    fi

    # Step 4: lazy suite, ffc succeeded and ran, no gfortran reference.
    if is_lazy_suite; then
        if check_xfail "$XFAIL_MANIFEST" "$rel_path"; then
            status="XPASS"
            note="listed in xfail manifest but ffc ran successfully"
            XPASS_COUNT=$((XPASS_COUNT + 1))
        else
            status="PASS"
            note="lazy suite, ffc ran successfully"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
            "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
        continue
    fi

    # Step 5: standard suite, compile with gfortran reference.
    if compile_with_gfortran "$full_path" "$ref_exe"; then
        ref_exit=0
    else
        ref_exit=1
    fi

    # Step 6: if gfortran failed, for standard Fortran this is unexpected
    # but some files may use non-standard features. Just check ffc exit code.
    if [ "$ref_exit" -ne 0 ]; then
        if check_xfail "$XFAIL_MANIFEST" "$rel_path"; then
            status="XFAIL"
            note="gfortran could not compile; listed in xfail"
            XFAIL_COUNT=$((XFAIL_COUNT + 1))
            printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
                "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
            continue
        else
            status="FAIL"
            note="gfortran could not compile reference; ffc_exit=$ffc_exit"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
                "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
            echo "  FAIL: $rel_path (gfortran failed, ffc_exit=$ffc_exit)"
            continue
        fi
    fi

    # Step 7: run gfortran reference
    run_capture "$ref_exe" "$ref_out" "$TIMEOUT"
    ref_exit=$?

    # Step 8: compare outputs.
    if compare_outputs "$ffc_out" "$ref_out" "$ffc_exit" "$ref_exit"; then
        if check_xfail "$XFAIL_MANIFEST" "$rel_path"; then
            status="XPASS"
            note="listed in xfail manifest but output matches gfortran"
            XPASS_COUNT=$((XPASS_COUNT + 1))
        else
            status="PASS"
            note="output matches gfortran"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
            "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
        continue
    fi

    # Step 9: mismatch, check xfail.
    if check_xfail "$XFAIL_MANIFEST" "$rel_path"; then
        status="XFAIL"
        note="output mismatch listed in xfail manifest"
        XFAIL_COUNT=$((XFAIL_COUNT + 1))
        printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
            "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
        continue
    fi

    status="FAIL"
    note="stdout or exit mismatch with gfortran"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    HAS_FAIL=1
    printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"}\n' \
        "$SUITE" "$(json_escape "$rel_path")" "$status" "$ffc_exit" "$ref_exit" "$(json_escape "$note")" >> "$REPORT"
    echo "  FAIL: $rel_path (output mismatch)"

done < "$FILE_LIST"

# Summary
printf '{"suite":"%s","status":"SUMMARY","pass":%d,"xfail":%d,"xpass":%d,"fail":%d}\n' \
    "$SUITE" "$PASS_COUNT" "$XFAIL_COUNT" "$XPASS_COUNT" "$FAIL_COUNT" >> "$REPORT"

echo ""
echo "=== $SUITE summary ==="
echo "  PASS=$PASS_COUNT  XFAIL=$XFAIL_COUNT  XPASS=$XPASS_COUNT  FAIL=$FAIL_COUNT  TOTAL=$TOTAL_COUNT"
echo "  Report: $REPORT"

# Exit nonzero only if non-xfail FAILs occurred
if [ "$HAS_FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
