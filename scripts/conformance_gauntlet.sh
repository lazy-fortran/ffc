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
#   --file PATH         select one suite-relative file (repeatable)
#   --files-from PATH   read suite-relative files from PATH (repeatable)
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
source "$SCRIPT_DIR/lib_expected_manifest.sh"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMARY_REPO_ROOT="$(resolve_primary_checkout_root "$PROJECT_DIR")"
CORPUS_PARENT="$(dirname "$PRIMARY_REPO_ROOT")"

# Defaults
SUITES="fortfront-f90 fortfront-lf lfortran gfortran-dg"
FFC_BIN=""
REPORT=""
MAX_FILES=""
TIMEOUT=5
HAS_FAIL=0
SELECTOR_KINDS=()
SELECTOR_VALUES=()

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

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
        --file)
            if [ $# -lt 2 ]; then fail "--file requires a path"; fi
            SELECTOR_KINDS+=("file")
            SELECTOR_VALUES+=("$2")
            shift 2 ;;
        --files-from)
            if [ $# -lt 2 ]; then fail "--files-from requires a path"; fi
            SELECTOR_KINDS+=("list")
            SELECTOR_VALUES+=("$2")
            shift 2 ;;
        --max-files)
            MAX_FILES="$2"; shift 2 ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        *)
            fail "unknown option $1" ;;
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
            echo "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/f90" ;;
        fortfront-lf)
            echo "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/lf" ;;
        lfortran)
            echo "${FFC_LFORTRAN_DIR:-$CORPUS_PARENT/lfortran}/integration_tests" ;;
        gfortran-dg)
            echo "${FFC_GFORTRAN_DG_DIR:-$CORPUS_PARENT/gcc/gcc/testsuite/gfortran.dg}" ;;
    esac
}

# Resolve xfail manifest
resolve_xfail_manifest() {
    local safe_suite
    safe_suite=${SUITE//-/_}
    echo "${FFC_XFAIL_MANIFEST:-$PROJECT_DIR/test/conformance/xfail_${safe_suite}.txt}"
}

resolve_skip_manifest() {
    local safe_suite
    safe_suite=${SUITE//-/_}
    echo "${FFC_SKIP_MANIFEST:-$PROJECT_DIR/test/conformance/skip_${safe_suite}.txt}"
}

resolve_undefined_output_manifest() {
    local safe_suite
    safe_suite=${SUITE//-/_}
    echo "${FFC_UNDEFINED_OUTPUT_MANIFEST:-$PROJECT_DIR/test/conformance/undefined_output_${safe_suite}.txt}"
}

# File extension for single-extension suites.
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

dg_skip_reason() {
    local source="$1"
    if grep -Eq 'dg-additional-sources' "$source"; then
        echo "multifile"
        return 0
    fi
    if dg_has_nonempty_options "$source"; then
        echo "flags"
        return 0
    fi
    if grep -Eq 'dg-(require|skip-if|final|prune-output|excess-errors|shouldfail)' "$source"; then
        echo "directive"
        return 0
    fi
    local dg_do
    dg_do=$(dg_do_mode "$source")
    case "$dg_do" in
        run|compile|"") return 1 ;;
        *) echo "directive"; return 0 ;;
    esac
}

dg_has_nonempty_options() {
    local source="$1" payload normalized
    while IFS= read -r payload; do
        normalized=$(printf '%s\n' "$payload" | tr -d '[:space:]"')
        if [ -n "$normalized" ]; then
            return 0
        fi
    done < <(sed -n 's/.*dg-\(add-\)\?options\([^}]*\)}.*/\2/p' "$source")
    return 1
}

dg_do_mode() {
    sed -n 's/.*dg-do[[:space:]]\+\([[:alnum:]_-]\+\).*/\1/p' "$1" | head -1
}

dg_test_kind() {
    local source="$1"
    if grep -Eq 'dg-error' "$source"; then
        echo "negative"
        return
    fi
    local dg_do
    dg_do=$(dg_do_mode "$source")
    case "$dg_do" in
        run) echo "run" ;;
        *) echo "compile" ;;
    esac
}

dg_warning_only() {
    local source="$1"
    grep -Eq 'dg-warning' "$source" && ! grep -Eq 'dg-error' "$source"
}

# Resolve ffc
if [ -z "$FFC_BIN" ]; then
    FFC_BIN=$(find_ffc) || exit 1
fi

normalize_manifest() {
    local src="$1" dst="$2"
    if [ ! -f "$src" ]; then
        : > "$dst"
        return
    fi
    sed 's/#.*$//' "$src" | \
        sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | \
        awk 'NF' > "$dst"
}

write_result_record() {
    local file="$1" result_status="$2" compiler_exit="$3" reference_exit="$4"
    local result_note="$5" warning_expectation="$6" warning_json=""
    if [ -n "$warning_expectation" ]; then
        warning_json=',"warning_expectation":"unchecked"'
    fi
    printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"%s}\n' \
        "$SUITE" "$(json_escape "$file")" "$result_status" "$compiler_exit" \
        "$reference_exit" "$(json_escape "$result_note")" "$warning_json" >> "$REPORT"
}

# Setup
export FFC_COMPILE_TIMEOUT="$TIMEOUT"
SUITE_ROOT=$(resolve_suite_root)
XFAIL_MANIFEST=$(resolve_xfail_manifest)
SKIP_MANIFEST=$(resolve_skip_manifest)
UNDEFINED_OUTPUT_MANIFEST=$(resolve_undefined_output_manifest)
EXT=$(file_extension)
TMPDIR_WORK=$(mktemp -d /tmp/ffc_gauntlet_XXXXXX)
trap 'rm -rf "$TMPDIR_WORK"' EXIT
XFAIL_LOOKUP="$TMPDIR_WORK/xfail_lookup.txt"
SKIP_LOOKUP="$TMPDIR_WORK/skip_lookup.txt"
UNDEFINED_OUTPUT_LOOKUP="$TMPDIR_WORK/undefined_output_lookup.txt"
validate_expected_manifest "$XFAIL_MANIFEST" "$XFAIL_LOOKUP" || exit 1
validate_expected_manifest "$SKIP_MANIFEST" "$SKIP_LOOKUP" || exit 1
normalize_manifest "$UNDEFINED_OUTPUT_MANIFEST" "$UNDEFINED_OUTPUT_LOOKUP"
manifest_overlap=$(grep -Fxf "$XFAIL_LOOKUP" "$UNDEFINED_OUTPUT_LOOKUP" || true)
if [ -n "$manifest_overlap" ]; then
    fail "files cannot be both xfail and undefined-output: $manifest_overlap"
fi
manifest_overlap=$(grep -Fxf "$SKIP_LOOKUP" "$UNDEFINED_OUTPUT_LOOKUP" || true)
if [ -n "$manifest_overlap" ]; then
    fail "files cannot be both skip and undefined-output: $manifest_overlap"
fi

# Counters
PASS_COUNT=0
XFAIL_COUNT=0
XPASS_COUNT=0
FAIL_COUNT=0
NOREF_COUNT=0
SKIP_COUNT=0
WARNING_UNCHECKED_COUNT=0
TOTAL_COUNT=0

# Clear report
> "$REPORT"

# Check suite root exists.
if [ ! -d "$SUITE_ROOT" ]; then
    echo "SKIP: $SUITE not found at $SUITE_ROOT"
    printf '{"suite":"%s","status":"SUMMARY","pass":%d,"xfail":%d,"xpass":%d,"fail":%d,"noref":%d,"skip":%d,"warning_unchecked":%d,"total":%d}\n' \
        "$SUITE" "$PASS_COUNT" "$XFAIL_COUNT" "$XPASS_COUNT" "$FAIL_COUNT" \
        "$NOREF_COUNT" "$SKIP_COUNT" "$WARNING_UNCHECKED_COUNT" "$TOTAL_COUNT" >> "$REPORT"
    exit 0
fi

# Collect files.
ALL_FILE_LIST="$TMPDIR_WORK/all_files.txt"
FILE_LIST="$TMPDIR_WORK/files.txt"
case "$SUITE" in
    fortfront-lf)
        find "$SUITE_ROOT" -maxdepth 1 \( -name "*.lf" -o -name "*.f90" \) -type f | sort > "$ALL_FILE_LIST" ;;
    *)
        find "$SUITE_ROOT" -maxdepth 1 -name "*.$EXT" -type f | sort > "$ALL_FILE_LIST" ;;
esac

SELECTED_LOOKUP="$TMPDIR_WORK/selected_files.txt"
: > "$SELECTED_LOOKUP"

add_selected_file() {
    local rel_path="$1"
    case "$rel_path" in
        "") fail "selected file path must not be empty" ;;
        /*) fail "selected file must be suite-relative: $rel_path" ;;
        ..|../*|*/../*|*/..) fail "selected file contains parent traversal: $rel_path" ;;
    esac
    if grep -Fqx -- "$rel_path" "$SELECTED_LOOKUP"; then
        fail "duplicate selected file: $rel_path"
    fi
    if ! grep -Fqx -- "$SUITE_ROOT/$rel_path" "$ALL_FILE_LIST"; then
        fail "unknown selected file: $rel_path"
    fi
    printf '%s\n' "$rel_path" >> "$SELECTED_LOOKUP"
    printf '%s\n' "$SUITE_ROOT/$rel_path" >> "$FILE_LIST"
}

if [ "${#SELECTOR_KINDS[@]}" -gt 0 ]; then
    : > "$FILE_LIST"
    for selector_index in "${!SELECTOR_KINDS[@]}"; do
        selector_kind=${SELECTOR_KINDS[$selector_index]}
        selector_value=${SELECTOR_VALUES[$selector_index]}
        if [ "$selector_kind" = "file" ]; then
            add_selected_file "$selector_value"
            continue
        fi
        if [ ! -f "$selector_value" ]; then
            fail "files-from path does not exist: $selector_value"
        fi
        while IFS= read -r selected_line || [ -n "$selected_line" ]; do
            selected_line=$(printf '%s\n' "$selected_line" | \
                sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
            [ -z "$selected_line" ] && continue
            case "$selected_line" in \#*) continue ;; esac
            add_selected_file "$selected_line"
        done < "$selector_value"
    done
else
    cp "$ALL_FILE_LIST" "$FILE_LIST"
fi

if [ "$MAX_FILES" -gt 0 ] 2>/dev/null; then
    head -n "$MAX_FILES" "$FILE_LIST" > "$TMPDIR_WORK/files_limited.txt"
    mv "$TMPDIR_WORK/files_limited.txt" "$FILE_LIST"
fi

FILE_COUNT=$(wc -l < "$FILE_LIST")
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "SKIP: no files found in $SUITE_ROOT for $SUITE"
    printf '{"suite":"%s","status":"SUMMARY","pass":%d,"xfail":%d,"xpass":%d,"fail":%d,"noref":%d,"skip":%d,"warning_unchecked":%d,"total":%d}\n' \
        "$SUITE" "$PASS_COUNT" "$XFAIL_COUNT" "$XPASS_COUNT" "$FAIL_COUNT" \
        "$NOREF_COUNT" "$SKIP_COUNT" "$WARNING_UNCHECKED_COUNT" "$TOTAL_COUNT" >> "$REPORT"
    exit 0
fi

# Build a module/submodule index of the suite directory so a file that USEs a
# module DEFINED in a sibling file can be compiled with separate compilation:
# the sibling files are compiled first into a per-test include dir and linked in.
# gfortran.dg models multifile cases through dg-additional-sources instead, so
# the index is built only for the flat source-tree suites.
MODULE_INDEX="$TMPDIR_WORK/module_index.tsv"
: > "$MODULE_INDEX"
if [ "$SUITE" != "gfortran-dg" ]; then
    build_module_index "$SUITE_ROOT" "$MODULE_INDEX"
fi

echo "Running $SUITE: $FILE_COUNT files, timeout=${TIMEOUT}s, ffc=$FFC_BIN"

# Process each file. The file list is read on FD 3, not stdin, so a compiled
# test program that reads stdin cannot consume the list and desynchronise the
# loop.
while IFS= read -r full_path <&3; do
    [ -z "$full_path" ] && continue
    TOTAL_COUNT=$((TOTAL_COUNT + 1))

    basename_file=$(basename "$full_path")
    # Suite-relative path is the basename for single-depth search
    rel_path="$basename_file"

    if check_xfail "$SKIP_LOOKUP" "$rel_path"; then
        SKIP_COUNT=$((SKIP_COUNT + 1))
        printf '{"suite":"%s","file":"%s","status":"SKIP","note":"listed in skip manifest"}\n' \
            "$SUITE" "$(json_escape "$rel_path")" >> "$REPORT"
        continue
    fi

    if [ "$SUITE" = "gfortran-dg" ]; then
        skip_reason=$(dg_skip_reason "$full_path") || skip_reason=""
        if [ -n "$skip_reason" ]; then
            status="FAIL"
            note="directive requires skip manifest entry: $skip_reason"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            printf '{"suite":"%s","file":"%s","status":"%s","note":"%s"}\n' \
                "$SUITE" "$(json_escape "$rel_path")" "$status" "$(json_escape "$note")" >> "$REPORT"
            echo "  FAIL: $rel_path (unlisted skip: $skip_reason)"
            continue
        fi
    fi

    ffc_exe="$TMPDIR_WORK/ffc_${TOTAL_COUNT}"
    ffc_obj="$TMPDIR_WORK/ffc_${TOTAL_COUNT}.o"
    ref_exe="$TMPDIR_WORK/ref_${TOTAL_COUNT}"
    ffc_out="$TMPDIR_WORK/ffc_out_${TOTAL_COUNT}"
    ref_out="$TMPDIR_WORK/ref_out_${TOTAL_COUNT}"

    rm -f "$ffc_exe" "$ffc_obj" "$ref_exe" "$ffc_out" "$ref_out"

    ffc_exit=-1
    ref_exit=-1
    status=""
    note=""
    warning_expectation=""

    if [ "$SUITE" = "gfortran-dg" ]; then
        dg_kind=$(dg_test_kind "$full_path")
        if dg_warning_only "$full_path"; then
            warning_expectation="unchecked"
            WARNING_UNCHECKED_COUNT=$((WARNING_UNCHECKED_COUNT + 1))
        fi
        if [ "$dg_kind" = "compile" ]; then
            if compile_object_with_ffc "$full_path" "$ffc_obj" "$FFC_BIN"; then
                ffc_exit=0
                if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
                    status="XPASS"
                    note="listed in xfail manifest but ffc -c succeeded"
                    XPASS_COUNT=$((XPASS_COUNT + 1))
                    echo "  XPASS: $rel_path (compile now succeeds)"
                else
                    status="PASS"
                    note="ffc -c succeeded"
                    PASS_COUNT=$((PASS_COUNT + 1))
                fi
            else
                ffc_exit=1
                if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
                    status="XFAIL"
                    note="listed in xfail manifest"
                    XFAIL_COUNT=$((XFAIL_COUNT + 1))
                else
                    status="FAIL"
                    note="ffc -c failed"
                    FAIL_COUNT=$((FAIL_COUNT + 1))
                    HAS_FAIL=1
                    echo "  FAIL: $rel_path (ffc -c failed)"
                fi
            fi
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            continue
        fi

        if [ "$dg_kind" = "negative" ]; then
            if compile_object_with_ffc "$full_path" "$ffc_obj" "$FFC_BIN"; then
                ffc_exit=0
            else
                ffc_exit=1
            fi
            if [ "$ffc_exit" -ne 0 ]; then
                if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
                    status="XPASS"
                    note="listed in xfail manifest but ffc rejected negative test"
                    XPASS_COUNT=$((XPASS_COUNT + 1))
                    echo "  XPASS: $rel_path (negative test now rejects)"
                else
                    status="PASS"
                    note="negative test rejected"
                    PASS_COUNT=$((PASS_COUNT + 1))
                fi
            else
                if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
                    status="XFAIL"
                    note="negative test accepted; listed in xfail manifest"
                    XFAIL_COUNT=$((XFAIL_COUNT + 1))
                else
                    status="FAIL"
                    note="negative test accepted by ffc"
                    FAIL_COUNT=$((FAIL_COUNT + 1))
                    HAS_FAIL=1
                    echo "  FAIL: $rel_path (negative test accepted)"
                fi
            fi
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            continue
        fi
    fi

    # Step 0: separate compilation. If this file USEs a module defined in a
    # sibling file, compile the prerequisite module/submodule files first into a
    # per-test include dir, then build this file with -I <dir> plus their object
    # files. The gfortran reference compiles the same sibling sources so its
    # binary links too. A self-contained file resolves to no prerequisites and
    # follows the single-file path unchanged. If any prerequisite fails to
    # compile with ffc, the module is unavailable and the main file falls
    # through to its normal failure handling below.
    ffc_extra=()
    ref_extra=()
    if [ -s "$MODULE_INDEX" ]; then
        prereq_list="$TMPDIR_WORK/prereq_${TOTAL_COUNT}.txt"
        resolve_prerequisites "$full_path" "$SUITE_ROOT" "$MODULE_INDEX" "$prereq_list"
        if [ -s "$prereq_list" ]; then
            inc_dir="$TMPDIR_WORK/inc_${TOTAL_COUNT}"
            mkdir -p "$inc_dir"
            ffc_extra=(-I "$inc_dir")
            # The gfortran reference ALWAYS receives the full prerequisite source
            # list so its binary links: whether ffc can compile a prerequisite is
            # an ffc concern, not the reference's. ffc objects are only added when
            # the prerequisite compiles; otherwise the main ffc build fails for
            # lack of the .fmod and is classified honestly below.
            prereq_idx=0
            while IFS= read -r prereq_src <&4; do
                [ -z "$prereq_src" ] && continue
                ref_extra+=("$prereq_src")
                prereq_obj="$inc_dir/prereq_${prereq_idx}.o"
                if compile_object_with_ffc_inc "$prereq_src" "$prereq_obj" \
                    "$FFC_BIN" "$inc_dir"; then
                    ffc_extra+=("$prereq_obj")
                fi
                prereq_idx=$((prereq_idx + 1))
            done 4< "$prereq_list"
        fi
    fi

    # Step 1: compile with ffc
    if compile_with_ffc "$full_path" "$ffc_exe" "$FFC_BIN" "${ffc_extra[@]}"; then
        ffc_exit=0
    else
        ffc_exit=1
    fi

    # Step 2: if ffc failed, classify immediately
    if [ "$ffc_exit" -ne 0 ]; then
        if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
            status="XFAIL"
            note="listed in xfail manifest"
            XFAIL_COUNT=$((XFAIL_COUNT + 1))
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            continue
        else
            status="FAIL"
            note="ffc compilation failed"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            echo "  FAIL: $rel_path (ffc failed)"
            continue
        fi
    fi

    # Step 3: run ffc binary
    run_capture "$ffc_exe" "$ffc_out" "$TIMEOUT"
    ffc_exit=$?

    # An ordinary nonzero exit (e.g. STOP 99) is a legitimate program result,
    # not a failure: defer judgement to the gfortran comparison in step 8.
    # Only a crash short-circuits here: timeout (124), loader/exec error
    # (126, 127), or a signal (>=128). Lazy suites have no reference, so any
    # nonzero exit stays a failure.
    if [ "$ffc_exit" -ne 0 ] && { [ "$ffc_exit" -ge 126 ] || \
        [ "$ffc_exit" -eq 124 ] || is_lazy_suite; }; then
        if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
            status="XFAIL"
            note="listed in xfail manifest (runtime failure)"
            XFAIL_COUNT=$((XFAIL_COUNT + 1))
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            continue
        else
            status="FAIL"
            note="ffc runtime failed (exit $ffc_exit)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            echo "  FAIL: $rel_path (runtime exit $ffc_exit)"
            continue
        fi
    fi

    # Step 4: lazy suite, ffc succeeded and ran, no gfortran reference.
    if is_lazy_suite; then
        if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
            status="XPASS"
            note="listed in xfail manifest but ffc ran successfully"
            XPASS_COUNT=$((XPASS_COUNT + 1))
            echo "  XPASS: $rel_path (lazy suite now runs)"
        else
            status="PASS"
            note="lazy suite, ffc ran successfully"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        continue
    fi

    # Step 5: standard suite, compile with gfortran reference. Prerequisite
    # sibling sources (if any) are compiled together so the reference links.
    if compile_with_gfortran "$full_path" "$ref_exe" "${ref_extra[@]}"; then
        ref_exit=0
    else
        ref_exit=1
    fi

    # Step 6: gfortran failed, but ffc already compiled and ran the file.
    if [ "$ref_exit" -ne 0 ]; then
        if check_xfail "$UNDEFINED_OUTPUT_LOOKUP" "$rel_path"; then
            status="FAIL"
            note="undefined-output reference failed to compile"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            echo "  FAIL: $rel_path (undefined-output reference failed)"
            continue
        fi
        NOREF_COUNT=$((NOREF_COUNT + 1))
        if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
            status="XPASS"
            note="listed in xfail manifest; gfortran rejects but ffc runs"
            XPASS_COUNT=$((XPASS_COUNT + 1))
            echo "  XPASS: $rel_path (gfortran rejects, ffc runs)"
        else
            status="PASS"
            note="gfortran rejects; ffc runs (NO-REF)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        continue
    fi

    # Step 7: run gfortran reference
    run_capture "$ref_exe" "$ref_out" "$TIMEOUT"
    ref_exit=$?

    if check_xfail "$UNDEFINED_OUTPUT_LOOKUP" "$rel_path"; then
        if [ "$ffc_exit" -eq 0 ] && [ "$ref_exit" -eq 0 ]; then
            status="PASS"
            note="undefined reference output; both executions completed"
            PASS_COUNT=$((PASS_COUNT + 1))
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            continue
        fi
        status="FAIL"
        note="undefined-output execution did not terminate normally"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HAS_FAIL=1
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        echo "  FAIL: $rel_path (undefined-output execution failed)"
        continue
    fi

    # Step 8: compare outputs.
    if compare_outputs "$ffc_out" "$ref_out" "$ffc_exit" "$ref_exit"; then
        if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
            status="XPASS"
            note="listed in xfail manifest but output matches gfortran"
            XPASS_COUNT=$((XPASS_COUNT + 1))
            echo "  XPASS: $rel_path (output now matches gfortran)"
        else
            status="PASS"
            note="output matches gfortran"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        continue
    fi

    # Step 8b: nondeterministic reference (CPU_TIME, SYSTEM_CLOCK, RANDOM_*).
    # When the reference itself differs between runs, an exact byte comparison
    # can never pass. Compare numeric structure instead: same tokens and text,
    # numeric magnitudes and field widths ignored.
    if [ "$SUITE" != "fortfront-lf" ] && \
        reference_is_nondeterministic "$ref_exe" "$TIMEOUT" && \
        compare_structural "$ffc_out" "$ref_out" "$ffc_exit" "$ref_exit"; then
        if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
            status="XPASS"
            note="listed in xfail manifest but structure matches nondeterministic gfortran"
            XPASS_COUNT=$((XPASS_COUNT + 1))
            echo "  XPASS: $rel_path (nondeterministic structure now matches)"
        else
            status="PASS"
            note="numeric structure matches nondeterministic gfortran"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        continue
    fi

    # Step 9: mismatch, check xfail.
    if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
        status="XFAIL"
        note="output mismatch listed in xfail manifest"
        XFAIL_COUNT=$((XFAIL_COUNT + 1))
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        continue
    fi

    status="FAIL"
    note="stdout or exit mismatch with gfortran"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    HAS_FAIL=1
    write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
        "$note" "$warning_expectation"
    echo "  FAIL: $rel_path (output mismatch)"

done 3< "$FILE_LIST"

# Summary
printf '{"suite":"%s","status":"SUMMARY","pass":%d,"xfail":%d,"xpass":%d,"fail":%d,"noref":%d,"skip":%d,"warning_unchecked":%d,"total":%d}\n' \
    "$SUITE" "$PASS_COUNT" "$XFAIL_COUNT" "$XPASS_COUNT" "$FAIL_COUNT" \
    "$NOREF_COUNT" "$SKIP_COUNT" "$WARNING_UNCHECKED_COUNT" "$TOTAL_COUNT" >> "$REPORT"

echo ""
echo "=== $SUITE summary ==="
echo "  PASS=$PASS_COUNT  XFAIL=$XFAIL_COUNT  XPASS=$XPASS_COUNT  FAIL=$FAIL_COUNT  NOREF=$NOREF_COUNT  SKIP=$SKIP_COUNT  WARNING_UNCHECKED=$WARNING_UNCHECKED_COUNT  TOTAL=$TOTAL_COUNT"
echo "  Report: $REPORT"

# Exit nonzero only if non-xfail FAILs occurred
if [ "$HAS_FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
