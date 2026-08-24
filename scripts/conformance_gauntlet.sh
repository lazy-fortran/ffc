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
#   --report PATH       JSONL report path
#                       (default: ${TMPDIR:-/tmp}/ffc_gauntlet_<suite>.jsonl)
#   --observations PATH immutable expectation-neutral JSONL observations
#                       (default: <report stem>.observations.jsonl)
#   --file PATH         select one suite-relative file (repeatable)
#   --files-from PATH   read suite-relative files from PATH (repeatable)
#   --max-files N       only test the first N files (for smoke runs)
#   --timeout N         per-file timeout in seconds (default: 5)
#   --repeat N          run the selection N times and merge; a file whose
#                       status differs between attempts is recorded FLAKY
#                       instead of taking the last result (default: 1)
#   --sample N          measure a deterministic random subset of N files
#                       instead of the whole suite. The report is marked
#                       sampled and full_run=false, so it can never be written
#                       into the checked-in parity snapshot. The summary
#                       carries the 95% confidence margin of the sampled rate.
#   --seed S            seed for --sample (default: 0); same seed and same
#                       corpus select exactly the same files
#   --ref-cache DIR     cache successful gfortran reference outputs under DIR,
#                       keyed by the complete execution descriptor. Reuse is exact:
#                       a cached comparison that does not match is discarded
#                       and the reference is rebuilt.
#   --require-provenance require clean inputs and a freshly built compiler
#
# Environment variables (suite roots):
#   FFC_FORTFRONT_DIR   default: ../fortfront
#   FFC_LFORTRAN_DIR    default: ../lfortran
#   FFC_GFORTRAN_DG_DIR default: ../gcc/gcc/testsuite/gfortran.dg
#   FFC_NOREF_MANIFEST  default: test/conformance/noref_<suite>.txt
#
# No foreign source files are copied into this repository.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_conformance.sh"
source "$SCRIPT_DIR/lib_expected_manifest.sh"
source "$SCRIPT_DIR/lib_conformance_observation.sh"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRIMARY_REPO_ROOT="$(resolve_primary_checkout_root "$PROJECT_DIR")"
CORPUS_PARENT="$(dirname "$PRIMARY_REPO_ROOT")"

# Defaults
SUITES="fortfront-f90 fortfront-lf lfortran gfortran-dg"
FFC_BIN=""
REPORT=""
OBSERVATIONS=""
MAX_FILES=""
TIMEOUT=5
REPEAT=1
SAMPLE_SIZE=""
SAMPLE_SEED=0
SAMPLE_POPULATION=0
SAMPLED=false
REF_CACHE_DIR=""
REF_CACHE_HITS=0
REF_CACHE_COMPILER=""
REQUIRE_PROVENANCE=0
HAS_FAIL=0
SELECTOR_KINDS=()
SELECTOR_VALUES=()

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

ORIGINAL_ARGS=("$@")

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
        --observations)
            [ $# -ge 2 ] || fail "--observations requires a path"
            OBSERVATIONS="$2"; shift 2 ;;
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
        --repeat)
            if [ $# -lt 2 ]; then fail "--repeat requires a count"; fi
            REPEAT="$2"; shift 2 ;;
        --sample)
            if [ $# -lt 2 ]; then fail "--sample requires a count"; fi
            SAMPLE_SIZE="$2"; shift 2 ;;
        --seed)
            if [ $# -lt 2 ]; then fail "--seed requires an integer"; fi
            SAMPLE_SEED="$2"; shift 2 ;;
        --ref-cache)
            if [ $# -lt 2 ]; then fail "--ref-cache requires a directory"; fi
            REF_CACHE_DIR="$2"; shift 2 ;;
        --require-provenance)
            REQUIRE_PROVENANCE=1; shift ;;
        *)
            fail "unknown option $1" ;;
    esac
done

case "$REPEAT" in
    ''|*[!0-9]*) fail "--repeat requires a positive integer" ;;
esac
if [ "$REPEAT" -lt 1 ]; then
    fail "--repeat requires a positive integer"
fi

if [ -n "$SAMPLE_SIZE" ]; then
    case "$SAMPLE_SIZE" in
        ''|*[!0-9]*) fail "--sample requires a positive integer" ;;
    esac
    if [ "$SAMPLE_SIZE" -lt 1 ]; then
        fail "--sample requires a positive integer"
    fi
fi
case "$SAMPLE_SEED" in
    ''|*[!0-9]*) fail "--seed requires a non-negative integer" ;;
esac

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
    REPORT="${TMPDIR:-/tmp}/ffc_gauntlet_${SUITE}.jsonl"
fi
if [ -z "$OBSERVATIONS" ]; then
    case "$REPORT" in
        *.jsonl) OBSERVATIONS="${REPORT%.jsonl}.observations.jsonl" ;;
        *) OBSERVATIONS="${REPORT}.observations.jsonl" ;;
    esac
fi
REPORT_CANONICAL=$(python3 -c \
    'import os, sys; print(os.path.realpath(sys.argv[1]))' "$REPORT") || \
    fail "cannot resolve --report path"
OBSERVATIONS_CANONICAL=$(python3 -c \
    'import os, sys; print(os.path.realpath(sys.argv[1]))' "$OBSERVATIONS") || \
    fail "cannot resolve --observations path"
if [ "$REPORT_CANONICAL" = "$OBSERVATIONS_CANONICAL" ]; then
    fail "--report and --observations must name different files"
fi

# Ensure output directories exist.
mkdir -p "$(dirname "$REPORT")" || fail "cannot create report directory"
mkdir -p "$(dirname "$OBSERVATIONS")" || \
    fail "cannot create observation directory"

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

resolve_noref_manifest() {
    local safe_suite
    safe_suite=${SUITE//-/_}
    echo "${FFC_NOREF_MANIFEST:-$PROJECT_DIR/test/conformance/noref_${safe_suite}.txt}"
}

# noref_category <suite_relative_path>
# Print the manifest category for a NOREF-classified file; return 1 otherwise.
noref_category() {
    local path="$1"
    awk -F '\t' -v target="$path" \
        '$1 == target { print $2; found = 1; exit } END { exit !found }' \
        "$NOREF_LOOKUP"
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

# lazy_negative_test <suite_relative_path>
# A FortFront lazy-mode example named error_* is a deliberately invalid source
# kept as a parser-rejection fixture. Its oracle is that ffc rejects it, so a
# nonzero compiler exit is the expected result and acceptance is the failure
# (#576). Without this the harness scored a correct rejection as FAIL.
lazy_negative_test() {
    case "$1" in
        error_*) return 0 ;;
        *) return 1 ;;
    esac
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

# A dg-error directive marked { xfail ... } records an error gfortran is not
# expected to emit, so it does not make the file a negative test.
dg_has_active_error() {
    grep -E 'dg-error' "$1" | grep -qv 'xfail'
}

dg_test_kind() {
    local source="$1"
    if dg_has_active_error "$source"; then
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
    grep -Eq 'dg-warning' "$source" && ! dg_has_active_error "$source"
}

# source_has_program_root <source>
# LFortran's integration_tests directory contains standalone companion sources
# (subroutines, functions and modules) beside their executable test roots.  A
# source without PROGRAM or BLOCK DATA cannot be run as a standalone binary.
source_has_program_root() {
    local source="$1"
    awk '
        {
            line = tolower($0)
            sub(/!.*/, "", line)
            if (line ~ /^[[:space:]]*[0-9]*[[:space:]]*program([[:space:]]|$)/ ||
                line ~ /^[[:space:]]*[0-9]*[[:space:]]*block[[:space:]]+data([[:space:]]|$)/) {
                found = 1
            }
        }
        END { exit !found }
    ' "$source"
}

# Resolve ffc
if [ "$REQUIRE_PROVENANCE" -eq 1 ]; then
    [ -z "$FFC_BIN" ] || fail "--require-provenance cannot be combined with --ffc"
    (cd "$PROJECT_DIR" && fo build) || fail "provenance build failed"
    FFC_BIN="$PROJECT_DIR/build/fo/bin/ffc"
elif [ -z "$FFC_BIN" ]; then
    FFC_BIN=$(find_ffc) || exit 1
fi

# classify_nonrunnable_noref <suite_relative_path> <source> <category>
# Cases with no behavioral oracle because the program is not a self-contained
# runnable unit. The reference must not build a complete executable: if it does,
# the file is a stable valid executable and the category does not apply.
classify_nonrunnable_noref() {
    local rel="$1" source="$2" category="$3"
    local obj="$TMPDIR_WORK/noref_${TOTAL_COUNT}.o"
    local exe="$TMPDIR_WORK/noref_ref_${TOTAL_COUNT}"
    local ffc_status=1 ref_status=1 record_note
    CASE_ACTION="compile-only"
    CASE_FFC_FLAGS="-c"
    CASE_REF_FLAGS="-w -J @private-module-dir"

    if compile_with_gfortran "$source" "$exe"; then
        set_last_action_evidence CASE_REF_COMPILE executed 0
        record_note="reference builds a runnable executable; $category not applicable"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HAS_FAIL=1
        write_result_record "$rel" "FAIL" "$ffc_status" 0 "$record_note" ""
        echo "  FAIL: $rel (noref category not applicable: $category)"
        return
    else
        ref_status=$?
        set_last_action_evidence CASE_REF_COMPILE executed "$ref_status"
    fi

    if compile_object_with_ffc "$source" "$obj" "$FFC_BIN"; then
        ffc_status=0
        set_last_action_evidence CASE_FFC_COMPILE executed 0
    else
        ffc_status=$?
        set_last_action_evidence CASE_FFC_COMPILE executed "$ffc_status"
    fi
    if [ "$category" = "compile-only" ] && [ "$ffc_status" -ne 0 ]; then
        record_note="compile-only noref case failed ffc -c"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HAS_FAIL=1
        write_result_record "$rel" "FAIL" "$ffc_status" "$ref_status" \
            "$record_note" ""
        echo "  FAIL: $rel (compile-only noref case failed ffc -c)"
        return
    fi

    IS_NOREF_RECORD=1
    NOREF_RECORD_REASON="$category"
    NOREF_COUNT=$((NOREF_COUNT + 1))
    PASS_COUNT=$((PASS_COUNT + 1))
    record_note="no behavioral oracle ($category)"
    write_result_record "$rel" "PASS" "$ffc_status" "$ref_status" \
        "$record_note" ""
}

write_result_record() {
    local file="$1" result_status="$2" compiler_exit="$3" reference_exit="$4"
    local result_note="$5" warning_expectation="$6" warning_json="" noref_json=""
    local noref_manifest_json="" phase diagnostic_sha crash_sha ffc_output_sha
    local ref_output_sha elapsed_ms ffc_compile_ms ffc_run_ms ref_compile_ms
    local ref_run_ms peak_rss compiler_flags_sha
    if [ -n "$warning_expectation" ]; then
        warning_json=',"warning_expectation":"unchecked"'
    fi
    if [ "$IS_NOREF_RECORD" -eq 1 ]; then
        noref_json=$(printf ',"noref":true,"noref_reason":"%s"' \
            "$(json_escape "$NOREF_RECORD_REASON")")
    fi
    if [ -n "$NOREF_MANIFEST_CATEGORY" ]; then
        noref_manifest_json=$(printf ',"noref_manifest_category":"%s"' \
            "$(json_escape "$NOREF_MANIFEST_CATEGORY")")
    fi
    phase=$(case_phase "$result_status" "$compiler_exit" "$reference_exit" \
        "$result_note")
    diagnostic_sha=$(case_diagnostic_signature)
    crash_sha=$(case_crash_signature "$compiler_exit" "$reference_exit")
    ffc_output_sha=$(sha256_file_or_empty "${ffc_out:-}")
    ref_output_sha=$(sha256_file_or_empty "${ref_out:-}")
    ffc_compile_ms=$(case_metric_ms ffc_compile)
    ffc_run_ms=$(case_metric_ms ffc_run)
    ref_compile_ms=$(case_metric_ms ref_compile)
    ref_run_ms=$(case_metric_ms ref_run)
    elapsed_ms=$((ffc_compile_ms + ffc_run_ms + ref_compile_ms + ref_run_ms + $(case_metric_ms dependency_compile)))
    peak_rss=$(case_peak_rss)
    compiler_flags_sha=$(printf 'ffc:%s\nref:%s\n' \
        "$CASE_FFC_FLAGS" "$CASE_REF_FLAGS" | sha256sum | cut -d ' ' -f 1)
    printf '{"suite":"%s","file":"%s","status":"%s","ffc_exit":%d,"ref_exit":%d,"note":"%s"%s%s%s,"epoch_sha256":"%s","action":"%s","ffc_compile_action":"%s","ffc_compile_exit":%d,"ffc_compile_termination":"%s","ffc_compile_signal":%d,"ffc_run_action":"%s","ffc_run_exit":%d,"ffc_run_termination":"%s","ffc_run_signal":%d,"ref_compile_action":"%s","ref_compile_exit":%d,"ref_compile_termination":"%s","ref_compile_signal":%d,"ref_run_action":"%s","ref_run_exit":%d,"ref_run_termination":"%s","ref_run_signal":%d,"source_sha256":"%s","dependency_closure_sha256":"%s","ffc_flags":"%s","ref_flags":"%s","compiler_flags_sha256":"%s","environment_sha256":"%s","target_triple":"%s","runtime_abi_sha256":"%s","harness_sha256":"%s","toolchain_sha256":"%s","phase":"%s","diagnostic_signature_sha256":"%s","crash_signature_sha256":"%s","ffc_output_sha256":"%s","ref_output_sha256":"%s","elapsed_ms":%d,"ffc_compile_ms":%d,"ffc_run_ms":%d,"ref_compile_ms":%d,"ref_run_ms":%d,"peak_rss_kb":%d,"semantic_tags":"%s","coverage_mode":"none","coverage_sha256":"%s"}\n' \
        "$SUITE" "$(json_escape "$file")" "$result_status" "$compiler_exit" \
        "$reference_exit" "$(json_escape "$result_note")" "$warning_json" \
        "$noref_json" "$noref_manifest_json" "$EPOCH_SHA256" "$CASE_ACTION" \
        "$CASE_FFC_COMPILE_ACTION" "$CASE_FFC_COMPILE_EXIT" \
        "$CASE_FFC_COMPILE_TERMINATION" "$CASE_FFC_COMPILE_SIGNAL" \
        "$CASE_FFC_RUN_ACTION" "$CASE_FFC_RUN_EXIT" \
        "$CASE_FFC_RUN_TERMINATION" "$CASE_FFC_RUN_SIGNAL" \
        "$CASE_REF_COMPILE_ACTION" "$CASE_REF_COMPILE_EXIT" \
        "$CASE_REF_COMPILE_TERMINATION" "$CASE_REF_COMPILE_SIGNAL" \
        "$CASE_REF_RUN_ACTION" "$CASE_REF_RUN_EXIT" \
        "$CASE_REF_RUN_TERMINATION" "$CASE_REF_RUN_SIGNAL" \
        "$CASE_SOURCE_SHA256" \
        "$(case_dependency_closure_sha256)" "$(json_escape "$CASE_FFC_FLAGS")" \
        "$(json_escape "$CASE_REF_FLAGS")" "$compiler_flags_sha" \
        "$ENVIRONMENT_SHA256" "$(json_escape "$TARGET_TRIPLE")" \
        "$RUNTIME_ABI_SHA256" "$HARNESS_SHA256" "$TOOLCHAIN_SHA256" \
        "$phase" "$diagnostic_sha" "$crash_sha" "$ffc_output_sha" \
        "$ref_output_sha" "$elapsed_ms" "$ffc_compile_ms" "$ffc_run_ms" \
        "$ref_compile_ms" "$ref_run_ms" "$peak_rss" \
        "$(json_escape "$CASE_SEMANTIC_TAGS")" "$EMPTY_SHA256" >> "$OBSERVATIONS"
}

sha256_file_or_empty() {
    local path="${1:-}"
    if [ -n "$path" ] && [ -f "$path" ]; then
        sha256sum "$path" | cut -d ' ' -f 1
    else
        printf '%s\n' "$EMPTY_SHA256"
    fi
}

digest_paths() {
    local path file label
    {
        for path in "$@"; do
            if [ -d "$path" ]; then
                while IFS= read -r file; do
                    label=${file#"$PROJECT_DIR"/}
                    printf '%s\t%s\n' "$label" "$(sha256_file_or_empty "$file")"
                done < <(find "$path" -type f -print | LC_ALL=C sort)
            elif [ -f "$path" ]; then
                label=${path#"$PROJECT_DIR"/}
                printf '%s\t%s\n' "$label" "$(sha256_file_or_empty "$path")"
            fi
        done
    } | LC_ALL=C sort -u | sha256sum | cut -d ' ' -f 1
}

declared_environment_sha256() {
    env | LC_ALL=C awk -F= '
        $1 == "PATH" || $1 == "LANG" || $1 == "LC_ALL" ||
        $1 == "LC_CTYPE" || $1 == "TZ" || $1 == "LIBRARY_PATH" ||
        $1 == "LD_LIBRARY_PATH" || $1 ~ /^GFORTRAN_/ ||
        $1 ~ /^FORTRAN_/ || $1 ~ /^OMP_/ || $1 ~ /^ACC_/ ||
        $1 ~ /^OBS_/ || ($1 ~ /^FFC_/ && $1 != "FFC_XFAIL_MANIFEST")
    ' | LC_ALL=C sort | sha256sum | cut -d ' ' -f 1
}

case_add_missing_dependency() {
    local suite_relative_path="$1"
    printf 'missing:suite:%s\t%s\n' "$suite_relative_path" "$EMPTY_SHA256" \
        >> "$CASE_DEPENDENCY_FILE"
}

case_snapshot_source() {
    local source="$1" include_dir stem_include_dir
    local -a snapshot_args=()
    shift
    stem_include_dir="${source%.*}"
    if [ -d "$stem_include_dir" ]; then
        snapshot_args+=(--include-dir "$stem_include_dir")
    fi
    for include_dir in "$@"; do
        snapshot_args+=(--include-dir "$include_dir")
    done
    python3 "$SCRIPT_DIR/conformance_source_snapshot.py" \
        --suite-root "$SUITE_ROOT" --destination "$CASE_SNAPSHOT_DIR" \
        --manifest "$CASE_DEPENDENCY_FILE" --status "$CASE_SNAPSHOT_STATUS" \
        "${snapshot_args[@]}" "$source"
}

case_dependency_closure_sha256() {
    LC_ALL=C sort -u "$CASE_DEPENDENCY_FILE" | sha256sum | cut -d ' ' -f 1
}

semantic_tags_for_source() {
    local source="$1" tags=""
    grep -Eiq '^[[:space:]]*module[[:space:]]' "$source" && tags="module"
    grep -Eiq '^[[:space:]]*submodule[[:space:]]' "$source" && tags="${tags:+$tags,}submodule"
    grep -Eiq '\b(coarray|sync[[:space:]]+(all|images|memory))\b' "$source" && tags="${tags:+$tags,}coarray"
    grep -Eiq '\b(do[[:space:]]+concurrent|forall)\b' "$source" && tags="${tags:+$tags,}parallel"
    grep -Eiq '\b(type|class)[[:space:]]*\(' "$source" && tags="${tags:+$tags,}derived-type"
    grep -Eiq '\b(interface|procedure)\b' "$source" && tags="${tags:+$tags,}procedure"
    printf '%s\n' "${tags:-none}"
}

initialize_case_provenance() {
    local source="$1"
    CASE_DEPENDENCY_FILE="$TMPDIR_WORK/dependencies_${TOTAL_COUNT}.tsv"
    CASE_SNAPSHOT_DIR="$TMPDIR_WORK/source_snapshot_${TOTAL_COUNT}"
    CASE_SNAPSHOT_STATUS="$TMPDIR_WORK/source_snapshot_${TOTAL_COUNT}.status"
    CONFORMANCE_METRICS_FILE="$TMPDIR_WORK/metrics_${TOTAL_COUNT}.tsv"
    FFC_COMPILER_DIAGNOSTIC_FILE="$TMPDIR_WORK/ffc_diagnostic_${TOTAL_COUNT}.txt"
    REF_COMPILER_DIAGNOSTIC_FILE="$TMPDIR_WORK/ref_diagnostic_${TOTAL_COUNT}.txt"
    export CONFORMANCE_METRICS_FILE FFC_COMPILER_DIAGNOSTIC_FILE
    export REF_COMPILER_DIAGNOSTIC_FILE
    : > "$CASE_DEPENDENCY_FILE"
    : > "$CASE_SNAPSHOT_STATUS"
    : > "$CONFORMANCE_METRICS_FILE"
    : > "$FFC_COMPILER_DIAGNOSTIC_FILE"
    : > "$REF_COMPILER_DIAGNOSTIC_FILE"
    CASE_SOURCE_PATH=$(case_snapshot_source "$source") || \
        fail "cannot snapshot source closure: $source"
    CASE_SOURCE_SHA256=$(sha256_file_or_empty "$CASE_SOURCE_PATH")
    CASE_SEMANTIC_TAGS=$(semantic_tags_for_source "$CASE_SOURCE_PATH")
    CASE_FFC_FLAGS="default"
    CASE_ACTION="compile-run"
    CASE_FFC_COMPILE_ACTION="not-run"
    CASE_FFC_COMPILE_EXIT=-1
    CASE_FFC_COMPILE_TERMINATION="not-run"
    CASE_FFC_COMPILE_SIGNAL=0
    CASE_FFC_RUN_ACTION="not-run"
    CASE_FFC_RUN_EXIT=-1
    CASE_FFC_RUN_TERMINATION="not-run"
    CASE_FFC_RUN_SIGNAL=0
    CASE_REF_COMPILE_ACTION="not-run"
    CASE_REF_COMPILE_EXIT=-1
    CASE_REF_COMPILE_TERMINATION="not-run"
    CASE_REF_COMPILE_SIGNAL=0
    CASE_REF_RUN_ACTION="not-run"
    CASE_REF_RUN_EXIT=-1
    CASE_REF_RUN_TERMINATION="not-run"
    CASE_REF_RUN_SIGNAL=0
    if is_lazy_suite; then
        CASE_REF_FLAGS="not-applicable"
    else
        CASE_REF_FLAGS="-w -J @private-module-dir"
    fi
}

action_termination() {
    local exit_status="$1"
    if [ "$exit_status" -eq -1 ]; then printf 'not-run\n'
    elif [ "$exit_status" -eq 124 ]; then printf 'timeout\n'
    elif [ "$exit_status" -eq 126 ] || [ "$exit_status" -eq 127 ]; then
        printf 'exec-error\n'
    elif [ "$exit_status" -ge 129 ]; then printf 'signal\n'
    else printf 'exit\n'; fi
}

action_signal() {
    local exit_status="$1"
    if [ "$exit_status" -eq 124 ]; then
        printf '15\n'
    elif [ "$exit_status" -ge 129 ]; then
        printf '%d\n' "$((exit_status - 128))"
    else
        printf '0\n'
    fi
}

set_inferred_action_evidence() {
    local prefix="$1" state="$2" exit_status="$3"
    printf -v "${prefix}_ACTION" '%s' "$state"
    printf -v "${prefix}_EXIT" '%d' "$exit_status"
    printf -v "${prefix}_TERMINATION" '%s' \
        "$(action_termination "$exit_status")"
    printf -v "${prefix}_SIGNAL" '%d' "$(action_signal "$exit_status")"
}

set_last_action_evidence() {
    local prefix="$1" state="$2" exit_status="$3"
    printf -v "${prefix}_ACTION" '%s' "$state"
    printf -v "${prefix}_EXIT" '%d' "$exit_status"
    printf -v "${prefix}_TERMINATION" '%s' \
        "$CONFORMANCE_ACTION_TERMINATION"
    printf -v "${prefix}_SIGNAL" '%d' "$CONFORMANCE_ACTION_SIGNAL"
}

canonical_flags() {
    local base="$1" arg rendered="$1"
    shift
    for arg in "$@"; do
        case "$arg" in
            "$TMPDIR_WORK"/*) arg="@case/${arg##*/}" ;;
            "$SUITE_ROOT"/*) arg="@suite/${arg#"$SUITE_ROOT"/}" ;;
            "$PROJECT_DIR"/*) arg="@ffc/${arg#"$PROJECT_DIR"/}" ;;
        esac
        rendered="$rendered $arg"
    done
    printf '%s\n' "$rendered"
}

case_metric_ms() {
    local label="$1"
    awk -F '\t' -v label="$label" '$1 == label { total += $2 * 1000 }
        END { printf "%.0f\n", total }' "$CONFORMANCE_METRICS_FILE"
}

case_peak_rss() {
    awk -F '\t' '$3 ~ /^[0-9]+$/ && $3 > peak { peak = $3 }
        END { printf "%d\n", peak }' "$CONFORMANCE_METRICS_FILE"
}

case_diagnostic_signature() {
    { cat "$FFC_COMPILER_DIAGNOSTIC_FILE" "$REF_COMPILER_DIAGNOSTIC_FILE"; } |
        sed -E -e "s#${TMPDIR_WORK}#@case#g" \
            -e 's#(/[^ /:]*)?/ffc_gfmod_[[:alnum:]]+#@ref-module#g' \
            -e 's#/tmp/liric_exe_obj_[[:alnum:]]+#@link-object#g' \
            -e 's/0x[0-9A-Fa-f]+/<addr>/g; s/\r$//' |
        sha256sum | cut -d ' ' -f 1
}

case_crash_signature() {
    local ffc_status="$1" ref_status="$2"
    if { [ "$ffc_status" -eq 124 ] || [ "$ffc_status" -ge 126 ]; } ||
        { [ "$ref_status" -eq 124 ] || [ "$ref_status" -ge 126 ]; }; then
        {
            printf 'ffc=%s ref=%s\n' "$ffc_status" "$ref_status"
            [ -f "${ffc_out:-}" ] && tail -n 20 "$ffc_out"
            [ -f "${ref_out:-}" ] && tail -n 20 "$ref_out"
        } | sed -E 's/0x[0-9A-Fa-f]+/<addr>/g; s/\r$//' |
            sha256sum | cut -d ' ' -f 1
    else
        printf '%s\n' "$EMPTY_SHA256"
    fi
}

case_phase() {
    local result_status="$1" ffc_status="$2" ref_status="$3" note="$4"
    if [ "$result_status" = "SKIP" ]; then printf 'skip\n'
    elif [[ "$note" == *directive* ]]; then printf 'directive\n'
    elif [[ "$note" == *runtime* ]]; then printf 'run\n'
    elif [[ "$note" == *mismatch* ]] || [[ "$note" == *matches* ]]; then
        printf 'compare\n'
    elif [[ "$note" == *gfortran* ]] || [[ "$note" == *reference* ]]; then
        printf 'reference\n'
    elif [ "$ffc_status" -ne 0 ]; then printf 'compile\n'
    elif [ "$ref_status" -ne 0 ]; then printf 'reference\n'
    elif [ -f "${ref_out:-}" ]; then printf 'compare\n'
    elif [ -f "${ffc_out:-}" ]; then printf 'run\n'
    else printf 'complete\n'; fi
}

# Reference-output cache. The key binds every declared source, tool, flag,
# environment, target, runtime ABI, corpus, and harness input. Any cached
# comparison that does not match is thrown away and remeasured (see step 7b).
reference_cache_key() {
    {
        printf 'cache_schema:2\n'
        printf 'compiler:%s\n' "$REF_CACHE_COMPILER"
        printf 'compiler_executable_sha256:%s\n' "$REFERENCE_COMPILER_SHA256"
        printf 'target:%s\n' "$TARGET_TRIPLE"
        printf 'environment_sha256:%s\n' "$ENVIRONMENT_SHA256"
        printf 'flags:%s\n' "$CASE_REF_FLAGS"
        printf 'runtime_abi_sha256:%s\n' "$RUNTIME_ABI_SHA256"
        printf 'harness_sha256:%s\n' "$HARNESS_SHA256"
        printf 'toolchain_sha256:%s\n' "$TOOLCHAIN_SHA256"
        printf 'policy:timeout=%s;skip=%s;noref=%s\n' "$TIMEOUT" \
            "$SKIP_MANIFEST_SHA256" "$NOREF_MANIFEST_SHA256"
        printf 'suite:%s\n' "$SUITE"
        printf 'corpus_revision:%s\ncorpus_tree:%s\n' \
            "$CORPUS_REVISION" "$CORPUS_TREE"
        printf 'source:%s\n' "$CASE_SOURCE_SHA256"
        printf 'dependency_closure:%s\n' "$(case_dependency_closure_sha256)"
        printf 'stdin:/dev/null\ncwd:empty-sandbox\n'
    } | sha256sum | cut -d ' ' -f 1
}

reference_cache_store() {
    local entry="$1" compile_status="$2" run_status="$3" out_file="$4"
    local temporary output_sha ready_tmp
    [ -n "$REF_CACHE_DIR" ] || return 0
    # Failures, timeouts, signals, and interrupted runs are observations, not
    # reusable reference results. Only a normal compile and exit 0 is cached.
    [ "$compile_status" -eq 0 ] && [ "$run_status" -eq 0 ] || return 0
    [ -f "$out_file" ] || return 0
    mkdir -p "$(dirname "$entry")" || return 0
    temporary=$(mktemp "$(dirname "$entry")/.cache_XXXXXX") || return 0
    ready_tmp=$(mktemp "$(dirname "$entry")/.ready_XXXXXX") || {
        rm -f "$temporary"; return 0; }
    cp "$out_file" "$temporary" || { rm -f "$temporary" "$ready_tmp"; return 0; }
    output_sha=$(sha256sum "$temporary" | cut -d ' ' -f 1)
    printf '2\t0\t0\t%s\n' "$output_sha" > "$ready_tmp"
    rm -f "$entry.ready"
    mv "$temporary" "$entry.out" || { rm -f "$temporary" "$ready_tmp"; return 0; }
    mv "$ready_tmp" "$entry.ready" || { rm -f "$ready_tmp"; return 0; }
}

reference_cache_discard() {
    local entry="$1"
    [ -n "$REF_CACHE_DIR" ] || return 0
    rm -f "$entry.ready" "$entry.out" "$entry.compile" "$entry.exit"
}

git_revision() {
    git -C "$1" rev-parse HEAD 2>/dev/null || \
        printf '%040d\n' 0
}

# Setup
export FFC_COMPILE_TIMEOUT="$TIMEOUT"
REFERENCE_COMPILER=$(gfortran --version 2>/dev/null | head -1)
[ -n "$REFERENCE_COMPILER" ] || REFERENCE_COMPILER="unknown"
REFERENCE_COMPILER_PATH=$(command -v gfortran 2>/dev/null || true)
if [ -n "$REF_CACHE_DIR" ]; then
    mkdir -p "$REF_CACHE_DIR" || fail "cannot create cache dir: $REF_CACHE_DIR"
    REF_CACHE_COMPILER="$REFERENCE_COMPILER"
fi
SUITE_ROOT=$(resolve_suite_root)
XFAIL_MANIFEST=$(resolve_xfail_manifest)
SKIP_MANIFEST=$(resolve_skip_manifest)
NOREF_MANIFEST=$(resolve_noref_manifest)
EXT=$(file_extension)
TMPDIR_WORK=$(mktemp -d "${TMPDIR:-/tmp}/ffc_gauntlet_XXXXXX")
OBSERVATION_DESTINATION="$OBSERVATIONS"
OBSERVATION_STAGING=""
cleanup_gauntlet() {
    if [ -n "${OBSERVATION_STAGING:-}" ]; then
        rm -f -- "$OBSERVATION_STAGING"
    fi
    rm -rf "$TMPDIR_WORK"
}
trap cleanup_gauntlet EXIT
OBSERVATION_STAGING=$(mktemp \
    "$(dirname "$OBSERVATION_DESTINATION")/.ffc_observation_XXXXXX") || \
    fail "cannot stage observation next to $OBSERVATION_DESTINATION"
OBSERVATIONS="$OBSERVATION_STAGING"
XFAIL_LOOKUP="$TMPDIR_WORK/xfail_lookup.txt"
SKIP_LOOKUP="$TMPDIR_WORK/skip_lookup.txt"
NOREF_LOOKUP="$TMPDIR_WORK/noref_lookup.tsv"
NOREF_PATHS="$TMPDIR_WORK/noref_paths.txt"
# Observation is expectation-blind. The XFAIL manifest is not opened until all
# selected cases have produced raw outcomes; an empty lookup keeps every
# compile, execution, reference, and oracle branch independent of expectations.
: > "$XFAIL_LOOKUP"
validate_expected_manifest "$SKIP_MANIFEST" "$SKIP_LOOKUP" || exit 1
validate_noref_manifest "$NOREF_MANIFEST" "$NOREF_LOOKUP" || exit 1
cut -f 1 "$NOREF_LOOKUP" > "$NOREF_PATHS"
manifest_overlap=$(grep -Fxf "$SKIP_LOOKUP" "$NOREF_PATHS" || true)
if [ -n "$manifest_overlap" ]; then
    fail "files cannot be both skip and noref: $manifest_overlap"
fi

# Counters
PASS_COUNT=0
XFAIL_COUNT=0
XPASS_COUNT=0
FAIL_COUNT=0
NOREF_COUNT=0
SKIP_COUNT=0
WARNING_UNCHECKED_COUNT=0
FLAKY_COUNT=0
TOTAL_COUNT=0
IS_NOREF_RECORD=0
NOREF_RECORD_REASON=""
NOREF_MANIFEST_CATEGORY=""

FFC_REVISION=$(git_revision "$PROJECT_DIR")
FFC_SOURCE_SHA256=$(ffc_source_sha256 "$PROJECT_DIR")
FFC_BINARY_SHA256=$(sha256sum "$FFC_BIN" | cut -d ' ' -f 1)
EMPTY_SHA256=$(printf '' | sha256sum | cut -d ' ' -f 1)
TARGET_TRIPLE=$(gfortran -dumpmachine 2>/dev/null || printf unknown)
ENVIRONMENT_SHA256=$(declared_environment_sha256)
HARNESS_SHA256=$(digest_paths \
    "$SCRIPT_DIR/conformance_gauntlet.sh" \
    "$SCRIPT_DIR/lib_conformance.sh" \
    "$SCRIPT_DIR/lib_expected_manifest.sh" \
    "$SCRIPT_DIR/lib_conformance_observation.sh" \
    "$SCRIPT_DIR/conformance_action.py" \
    "$SCRIPT_DIR/conformance_source_snapshot.py" \
    "$SCRIPT_DIR/conformance_observation.py")
RUNTIME_ABI_SHA256=$(digest_paths \
    "$PROJECT_DIR/docs/RUNTIME_ABI.md" "$PROJECT_DIR/runtime" \
    "$PROJECT_DIR/src/ffc_runtime_source.f90" \
    "$PROJECT_DIR/src/ffc_runtime_link.f90" \
    "$PROJECT_DIR/src/liric_session_runtime_bindings.f90")
REFERENCE_COMPILER_SHA256=$(sha256_file_or_empty "$REFERENCE_COMPILER_PATH")
TOOLCHAIN_SHA256=$({
    printf 'ffc=%s\nreference=%s\nversion=%s\ntarget=%s\n' \
        "$FFC_BINARY_SHA256" "$REFERENCE_COMPILER_SHA256" \
        "$REFERENCE_COMPILER" "$TARGET_TRIPLE"
} | sha256sum | cut -d ' ' -f 1)
GLOBAL_COMPILER_FLAGS_SHA256=$(printf '%s\n' \
    'ffc:default;reference:-w -J @private-module-dir' | \
    sha256sum | cut -d ' ' -f 1)
FORTFRONT_REVISION=$(git_revision "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}")
LIRIC_REVISION=$(git_revision "${FFC_LIRIC_DIR:-$CORPUS_PARENT/liric}")
CORPUS_REVISION=$(git_revision "$SUITE_ROOT")
FORTFRONT_TREE=$(git_tree_revision "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}" || printf '%040d\n' 0)
LIRIC_TREE=$(git_tree_revision "${FFC_LIRIC_DIR:-$CORPUS_PARENT/liric}" || printf '%040d\n' 0)
CORPUS_TREE=$(git_tree_revision "$SUITE_ROOT" || printf '%040d\n' 0)
CORPUS_FILES_SHA256=$(printf '' | sha256sum | cut -d ' ' -f 1)
if [ -f "$SKIP_MANIFEST" ]; then
    SKIP_MANIFEST_SHA256=$(sha256sum "$SKIP_MANIFEST" | cut -d ' ' -f 1)
else
    SKIP_MANIFEST_SHA256=$EMPTY_SHA256
fi
if [ -f "$NOREF_MANIFEST" ]; then
    NOREF_MANIFEST_SHA256=$(sha256sum "$NOREF_MANIFEST" | cut -d ' ' -f 1)
else
    NOREF_MANIFEST_SHA256=$EMPTY_SHA256
fi
# ffc #642: identical commits built in two worktrees have been observed to
# disagree on corpus results. Every report names the checkout that produced it,
# so a cross-worktree comparison is visibly invalid instead of silently wrong.
WORKTREE_ID=$(python3 -c \
    'import os, sys; print(os.path.realpath(sys.argv[1]))' "$PROJECT_DIR") || \
    fail "cannot resolve ffc worktree path"
PROVENANCE_VERIFIED=false
FULL_RUN=true

compute_epoch_sha256() {
    local selection_sha256="${1:-$EMPTY_SHA256}"
    {
        printf 'epoch_schema:2\n'
        printf 'suite:%s\nselection:%s\ncorpus:%s:%s:%s\n' "$SUITE" \
            "$selection_sha256" "$CORPUS_REVISION" "$CORPUS_TREE" \
            "$CORPUS_FILES_SHA256"
        printf 'ffc:%s:%s:%s\n' "$FFC_REVISION" "$FFC_SOURCE_SHA256" \
            "$FFC_BINARY_SHA256"
        printf 'fortfront:%s:%s\nliric:%s:%s\n' "$FORTFRONT_REVISION" \
            "$FORTFRONT_TREE" "$LIRIC_REVISION" "$LIRIC_TREE"
        printf 'target:%s\nenvironment:%s\nruntime:%s\nharness:%s\ntoolchain:%s\n' \
            "$TARGET_TRIPLE" "$ENVIRONMENT_SHA256" "$RUNTIME_ABI_SHA256" \
            "$HARNESS_SHA256" "$TOOLCHAIN_SHA256"
        printf 'flags:%s\ntimeout:%s\nskip:%s\nnoref:%s\n' \
            "$GLOBAL_COMPILER_FLAGS_SHA256" "$TIMEOUT" \
            "$SKIP_MANIFEST_SHA256" "$NOREF_MANIFEST_SHA256"
        printf 'cache:%s\nfull_run:%s\nworktree:%s\n' \
            "$([ -n "$REF_CACHE_DIR" ] && printf enabled || printf disabled)" \
            "$FULL_RUN" "$WORKTREE_ID"
    } | sha256sum | cut -d ' ' -f 1
}

EPOCH_SHA256=$(compute_epoch_sha256 "$EMPTY_SHA256")
if [ "${#SELECTOR_KINDS[@]}" -gt 0 ] || [ "${MAX_FILES:-0}" -gt 0 ] 2>/dev/null; then
    FULL_RUN=false
fi
if [ -n "$SAMPLE_SIZE" ]; then
    SAMPLED=true
    FULL_RUN=false
fi

# sample_margin_pct <observed> <sample> <population>
# Half-width of the 95% confidence interval for the observed rate, in percent,
# with the finite-population correction. A sampled figure is never an exact
# one, so every sampled report states the margin next to the rate.
sample_margin_pct() {
    awk -v observed="$1" -v n="$2" -v pop="$3" 'BEGIN {
        if (n <= 0 || pop <= 1 || n >= pop) { printf "0.0\n"; exit }
        p = observed / n
        fpc = (pop - n) / (pop - 1)
        margin = 1.96 * sqrt(p * (1 - p) / n * fpc) * 100
        printf "%.1f\n", margin
    }'
}

echo_sample_line() {
    local margin rate
    [ "$SAMPLED" = true ] || return 0
    margin=$(sample_margin_pct "$PASS_COUNT" "$TOTAL_COUNT" \
        "$SAMPLE_POPULATION")
    rate=$(awk -v p="$PASS_COUNT" -v n="$TOTAL_COUNT" 'BEGIN {
        if (n <= 0) { printf "0.0\n"; exit }
        printf "%.1f\n", 100 * p / n
    }')
    echo "  SAMPLED: $TOTAL_COUNT of $SAMPLE_POPULATION files (seed $SAMPLE_SEED); PASS rate ${rate}% +/- ${margin}% (95% CI)"
    echo "  SAMPLED: estimate only; not valid for the parity snapshot"
}

write_summary() {
    local flaky_json="" sample_json="" margin
    if [ "$FLAKY_COUNT" -gt 0 ]; then
        flaky_json=$(printf ',"flaky":%d' "$FLAKY_COUNT")
    fi
    if [ "$SAMPLED" = true ]; then
        margin=$(sample_margin_pct "$PASS_COUNT" "$TOTAL_COUNT" \
            "$SAMPLE_POPULATION")
        sample_json=$(printf \
            ',"sampled":true,"sample_size":%d,"sample_population":%d,"sample_seed":%d,"sample_margin_pct":"%s"' \
            "$TOTAL_COUNT" "$SAMPLE_POPULATION" "$SAMPLE_SEED" "$margin")
    fi
    flaky_json="${flaky_json}${sample_json}"
    printf '{"suite":"%s","status":"SUMMARY","pass":%d,"xfail":%d,"xpass":%d,"fail":%d,"noref":%d,"skip":%d,"warning_unchecked":%d,"total":%d,"schema_version":2,"full_run":%s,"provenance_verified":%s,"epoch_sha256":"%s","ffc_revision":"%s","ffc_source_sha256":"%s","ffc_binary_sha256":"%s","fortfront_revision":"%s","fortfront_tree":"%s","liric_revision":"%s","liric_tree":"%s","corpus_revision":"%s","corpus_tree":"%s","corpus_files_sha256":"%s","worktree":"%s","report_kind":"observation","observation_schema_version":2,"reference_compiler":"%s","reference_cache_enabled":%s,"reference_cache_hits":%d,"timeout_seconds":%d,"skip_manifest_sha256":"%s","noref_manifest_sha256":"%s","target_triple":"%s","environment_sha256":"%s","runtime_abi_sha256":"%s","harness_sha256":"%s","toolchain_sha256":"%s","compiler_flags_sha256":"%s","coverage_mode":"none"%s}\n' \
        "$SUITE" "$PASS_COUNT" "$XFAIL_COUNT" "$XPASS_COUNT" "$FAIL_COUNT" \
        "$NOREF_COUNT" "$SKIP_COUNT" "$WARNING_UNCHECKED_COUNT" "$TOTAL_COUNT" \
        "$FULL_RUN" "$PROVENANCE_VERIFIED" "$EPOCH_SHA256" "$FFC_REVISION" \
        "$FFC_SOURCE_SHA256" "$FFC_BINARY_SHA256" "$FORTFRONT_REVISION" \
        "$FORTFRONT_TREE" "$LIRIC_REVISION" "$LIRIC_TREE" \
        "$CORPUS_REVISION" "$CORPUS_TREE" "$CORPUS_FILES_SHA256" \
        "$(json_escape "$WORKTREE_ID")" \
        "$(json_escape "$REFERENCE_COMPILER")" \
        "$([ -n "$REF_CACHE_DIR" ] && printf true || printf false)" \
        "$REF_CACHE_HITS" "$TIMEOUT" "$SKIP_MANIFEST_SHA256" \
        "$NOREF_MANIFEST_SHA256" "$(json_escape "$TARGET_TRIPLE")" \
        "$ENVIRONMENT_SHA256" "$RUNTIME_ABI_SHA256" "$HARNESS_SHA256" \
        "$TOOLCHAIN_SHA256" "$GLOBAL_COMPILER_FLAGS_SHA256" \
        "$flaky_json" >> "$OBSERVATIONS"
}

publish_observation() {
    if ! conformance_observation_publish "$OBSERVATIONS" \
            "$OBSERVATION_DESTINATION" "$SUITE"; then
        printf 'ERROR: refusing to publish incomplete observation: %s\n' \
            "$OBSERVATIONS" >&2
        return 1
    fi
    rm -f -- "$OBSERVATIONS"
    OBSERVATION_STAGING=""
    OBSERVATIONS="$OBSERVATION_DESTINATION"
}

# Materialize the default expectation view only after observation is complete.
# A nonzero result can mean either classified FAIL/FLAKY records or a malformed
# expectation manifest; in both cases the raw observation remains available.
classify_observation_report() {
    local classification_status=0 counters

    conformance_observation_classify "$OBSERVATIONS" "$REPORT" "$SUITE" \
        "$XFAIL_MANIFEST" manifest || classification_status=$?
    if [ "$classification_status" -gt 1 ]; then
        printf 'ERROR: classification failed; observations preserved at %s\n' \
            "$OBSERVATIONS" >&2
        return "$classification_status"
    fi
    counters=$(conformance_observation_classification_counts \
        "$REPORT" "$SUITE") || return 2
    IFS=$'\t' read -r PASS_COUNT XFAIL_COUNT XPASS_COUNT FAIL_COUNT \
        FLAKY_COUNT NOREF_COUNT SKIP_COUNT WARNING_UNCHECKED_COUNT \
        TOTAL_COUNT SAMPLE_POPULATION <<< "$counters"
    return "$classification_status"
}

# Repeated runs: a single run cannot distinguish a stable result from a case
# that flips between attempts. Each attempt is a fully independent child run
# through the same classification path; the merge below records a file whose
# status is not identical in every attempt as FLAKY, so nothing downstream can
# read an intermittent pass as a stable one.
run_repeated_attempts() {
    local attempt child_args=() attempt_observations=() skip_next=0 arg
    local child_status
    local attempt_dir classification_status=0

    for arg in "${ORIGINAL_ARGS[@]}"; do
        if [ "$skip_next" -eq 1 ]; then
            skip_next=0
            continue
        fi
        case "$arg" in
            --repeat|--report|--observations) skip_next=1 ;;
            *) child_args+=("$arg") ;;
        esac
    done

    attempt_dir="$TMPDIR_WORK/attempts"
    mkdir -p "$attempt_dir"
    for attempt in $(seq 1 "$REPEAT"); do
        child_status=0
        echo "=== attempt $attempt/$REPEAT ==="
        attempt_observations+=("$attempt_dir/attempt_${attempt}.observations.jsonl")
        bash "$0" "${child_args[@]}" --repeat 1 \
            --observations "$attempt_dir/attempt_${attempt}.observations.jsonl" \
            --report "$attempt_dir/attempt_${attempt}.jsonl" || child_status=$?
        if ! conformance_observation_validate \
                "$attempt_dir/attempt_${attempt}.observations.jsonl" "$SUITE"; then
            printf 'ERROR: repeat attempt %d produced no complete observation (exit %d)\n' \
                "$attempt" "$child_status" >&2
            return 1
        fi
    done

    conformance_observation_merge "$OBSERVATIONS" "$SUITE" \
        "${attempt_observations[@]}" || return 1
    publish_observation || return 1
    classify_observation_report || classification_status=$?
    [ "$classification_status" -le 1 ] || return "$classification_status"

    echo ""
    echo "=== $SUITE summary (merged over $REPEAT attempts) ==="
    echo "  PASS=$PASS_COUNT  XFAIL=$XFAIL_COUNT  XPASS=$XPASS_COUNT  FAIL=$FAIL_COUNT  FLAKY=$FLAKY_COUNT  NOREF=$NOREF_COUNT  SKIP=$SKIP_COUNT  TOTAL=$TOTAL_COUNT"
    echo_sample_line
    echo "  Report: $REPORT"
    echo "  Observations: $OBSERVATIONS"

    return "$classification_status"
}

if [ "$REPEAT" -gt 1 ]; then
    run_repeated_attempts
    exit $?
fi

# Check suite root exists.
if [ ! -d "$SUITE_ROOT" ]; then
    echo "SKIP: $SUITE not found at $SUITE_ROOT"
    write_summary
    publish_observation || exit 1
    classification_status=0
    classify_observation_report || classification_status=$?
    exit "$classification_status"
fi

if [ "$REQUIRE_PROVENANCE" -eq 1 ]; then
    FORTFRONT_DIR=${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}
    LIRIC_DIR=${FFC_LIRIC_DIR:-$CORPUS_PARENT/liric}
    require_clean_git_tree "$PROJECT_DIR" ffc src app fpm.toml || exit 1
    require_clean_git_tree "$FORTFRONT_DIR" FortFront || exit 1
    require_clean_git_tree "$LIRIC_DIR" LIRIC || exit 1
    require_clean_git_tree "$SUITE_ROOT" "$SUITE corpus" || exit 1
    require_compiler_inputs_older_than_binary "$FFC_BIN" "$PROJECT_DIR" \
        "$FORTFRONT_DIR" "$LIRIC_DIR" || exit 1
    PROVENANCE_VERIFIED=true
fi

# Collect files.
ALL_FILE_LIST="$TMPDIR_WORK/all_files.txt"
FILE_LIST="$TMPDIR_WORK/files.txt"
# Collate in C so the corpus order, and the digest taken over it below, do not
# depend on the caller's locale. suite_files_sha256 in lib_parity_dashboard.sh
# hashes the same list and must agree with it.
case "$SUITE" in
    fortfront-lf)
        find "$SUITE_ROOT" -maxdepth 1 \( -name "*.lf" -o -name "*.f90" \) -type f | LC_ALL=C sort > "$ALL_FILE_LIST" ;;
    *)
        find "$SUITE_ROOT" -maxdepth 1 -name "*.$EXT" -type f | LC_ALL=C sort > "$ALL_FILE_LIST" ;;
esac
CORPUS_FILES_SHA256=$(sed "s#^$SUITE_ROOT/##" "$ALL_FILE_LIST" | \
    sha256sum | cut -d ' ' -f 1)

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

# Stratified sampling happens per suite: each suite draws its own subset, so
# every suite keeps its own margin. The draw is a deterministic function of the
# seed and the file path (FNV-1a over "seed:path"), so the same seed over the
# same corpus selects exactly the same files on any machine and in any locale,
# and the selection stays in corpus order in the report.
draw_sample() {
    local size="$1" seed="$2" src="$3" dest="$4"
    awk -v seed="$seed" '
        function fnv(text,    i, hash) {
            hash = 2166136261
            for (i = 1; i <= length(text); i++) {
                hash = xor_byte(hash, index(CHARS, substr(text, i, 1)) + 31)
                hash = (hash * 16777619) % 4294967296
            }
            return hash
        }
        function xor_byte(value, byte,    i, bit, result, v, b) {
            result = 0
            v = value % 256
            b = byte % 256
            for (i = 0; i < 8; i++) {
                bit = (int(v / 2 ^ i) % 2) != (int(b / 2 ^ i) % 2)
                if (bit) result += 2 ^ i
            }
            return value - v + result
        }
        BEGIN {
            CHARS = " !\"#$%&'"'"'()*+,-./0123456789:;<=>?@" \
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`" \
                "abcdefghijklmnopqrstuvwxyz{|}~"
        }
        { printf "%010d\t%s\n", fnv(seed ":" $0), $0 }
    ' "$src" | LC_ALL=C sort | head -n "$size" | cut -f 2- | LC_ALL=C sort \
        > "$dest"
}

if [ -n "$SAMPLE_SIZE" ]; then
    SAMPLE_POPULATION=$(wc -l < "$FILE_LIST")
    if [ "$SAMPLE_SIZE" -lt "$SAMPLE_POPULATION" ]; then
        draw_sample "$SAMPLE_SIZE" "$SAMPLE_SEED" "$FILE_LIST" \
            "$TMPDIR_WORK/files_sampled.txt"
        mv "$TMPDIR_WORK/files_sampled.txt" "$FILE_LIST"
    else
        SAMPLE_SIZE="$SAMPLE_POPULATION"
    fi
fi

FILE_COUNT=$(wc -l < "$FILE_LIST")
SELECTION_SHA256=$(sed "s#^$SUITE_ROOT/##" "$FILE_LIST" | \
    sha256sum | cut -d ' ' -f 1)
EPOCH_SHA256=$(compute_epoch_sha256 "$SELECTION_SHA256")
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "SKIP: no files found in $SUITE_ROOT for $SUITE"
    write_summary
    publish_observation || exit 1
    classification_status=0
    classify_observation_report || classification_status=$?
    exit "$classification_status"
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
    IS_NOREF_RECORD=0
    NOREF_RECORD_REASON=""
    NOREF_MANIFEST_CATEGORY=""

    basename_file=$(basename "$full_path")
    # Suite-relative path is the basename for single-depth search
    rel_path="$basename_file"
    ffc_out=""
    ref_out=""
    initialize_case_provenance "$full_path"
    full_path="$CASE_SOURCE_PATH"

    if check_xfail "$SKIP_LOOKUP" "$rel_path"; then
        CASE_ACTION="exclude"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        write_result_record "$rel_path" "SKIP" -1 -1 \
            "listed in skip manifest" ""
        continue
    fi

    if [ -s "$CASE_SNAPSHOT_STATUS" ]; then
        CASE_ACTION="exclude"
        status="FAIL"
        note=$(head -n 1 "$CASE_SNAPSHOT_STATUS")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HAS_FAIL=1
        write_result_record "$rel_path" "$status" -1 -1 "$note" ""
        echo "  FAIL: $rel_path ($note)"
        continue
    fi

    if [ "$SUITE" = "lfortran" ] && ! source_has_program_root "$full_path"; then
        CASE_ACTION="exclude"
        SKIP_COUNT=$((SKIP_COUNT + 1))
        write_result_record "$rel_path" "SKIP" -1 -1 \
            "no PROGRAM or BLOCK DATA root; standalone executable is not applicable" ""
        echo "  SKIP: $rel_path (no PROGRAM or BLOCK DATA root; standalone executable is not applicable)"
        continue
    fi

    noref_kind=$(noref_category "$rel_path") || noref_kind=""
    NOREF_MANIFEST_CATEGORY="$noref_kind"
    if [ -n "$noref_kind" ] && [ "$noref_kind" != "undefined-runtime-value" ] && \
        [ "$noref_kind" != "nondeterministic-runtime-value" ]; then
        classify_nonrunnable_noref "$rel_path" "$full_path" "$noref_kind"
        continue
    fi

    if [ "$SUITE" = "gfortran-dg" ]; then
        skip_reason=$(dg_skip_reason "$full_path") || skip_reason=""
        if [ -n "$skip_reason" ]; then
            CASE_ACTION="exclude"
            status="FAIL"
            note="directive requires skip manifest entry: $skip_reason"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            write_result_record "$rel_path" "$status" -1 -1 "$note" ""
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
            CASE_ACTION="compile-only"
            CASE_FFC_FLAGS="-c"
            CASE_REF_FLAGS="not-run"
            if compile_object_with_ffc "$full_path" "$ffc_obj" "$FFC_BIN"; then
                ffc_exit=0
                set_last_action_evidence CASE_FFC_COMPILE executed 0
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
                ffc_exit=$?
                set_last_action_evidence CASE_FFC_COMPILE executed \
                    "$ffc_exit"
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
            CASE_ACTION="reject"
            CASE_FFC_FLAGS="-c"
            CASE_REF_FLAGS="not-run"
            if compile_object_with_ffc "$full_path" "$ffc_obj" "$FFC_BIN"; then
                ffc_exit=0
                set_last_action_evidence CASE_FFC_COMPILE executed 0
            else
                ffc_exit=$?
                set_last_action_evidence CASE_FFC_COMPILE executed \
                    "$ffc_exit"
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
    prereq_sources=()
    extra_sources=()
    inc_dir=""

    # LFortran's integration tests keep INCLUDE material in a directory named
    # after the test (CMake INCLUDE_PATH <stem>). Both compilers get that
    # directory on their include search path so the reference and ffc see the
    # same source.
    stem_include_dir="${full_path%.*}"
    if [ -d "$stem_include_dir" ]; then
        ffc_extra=(-I "$stem_include_dir")
        ref_extra=(-I "$stem_include_dir")
    fi
    if [ -s "$MODULE_INDEX" ]; then
        prereq_list="$TMPDIR_WORK/prereq_${TOTAL_COUNT}.txt"
        resolve_prerequisites "$full_path" "$SUITE_ROOT" "$MODULE_INDEX" "$prereq_list"
        if [ -s "$prereq_list" ]; then
            inc_dir="$TMPDIR_WORK/inc_${TOTAL_COUNT}"
            mkdir -p "$inc_dir"
            ffc_extra+=(-I "$inc_dir")
            while IFS= read -r prereq_src <&4; do
                [ -z "$prereq_src" ] && continue
                prereq_src=$(case_snapshot_source "$prereq_src") || \
                    fail "cannot snapshot prerequisite closure: $prereq_src"
                prereq_sources+=("$prereq_src")
                ref_extra+=("$prereq_src")
            done 4< "$prereq_list"
        fi
    fi

    # Some LFortran integration tests name non-module companion sources in
    # CMake EXTRAFILES. Keep that harness contract explicit and bounded rather
    # than silently treating a link failure as an implementation failure.
    extra_manifest="$PROJECT_DIR/test/conformance/extra_${SUITE}.txt"
    extra_list="$TMPDIR_WORK/extra_${TOTAL_COUNT}.txt"
    missing_extra_source=""
    resolve_extra_sources "$rel_path" "$extra_manifest" > "$extra_list"
    if [ -s "$extra_list" ]; then
        if [ -z "${inc_dir:-}" ]; then
            inc_dir="$TMPDIR_WORK/inc_${TOTAL_COUNT}"
            mkdir -p "$inc_dir"
            ffc_extra+=(-I "$inc_dir")
        fi
        extra_idx=0
        while IFS= read -r extra_name; do
            [ -z "$extra_name" ] && continue
            extra_src="$SUITE_ROOT/$extra_name"
            if [ ! -f "$extra_src" ]; then
                missing_extra_source="$extra_name"
                case_add_missing_dependency "$extra_name"
                break
            fi
            extra_src=$(case_snapshot_source "$extra_src") || \
                fail "cannot snapshot extra-source closure: $extra_src"
            extra_sources+=("$extra_src")
            ref_extra+=("$extra_src")
            extra_idx=$((extra_idx + 1))
        done < "$extra_list"
    fi
    if [ -n "$missing_extra_source" ]; then
        status="FAIL"
        note="missing extra source $missing_extra_source"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HAS_FAIL=1
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        echo "  FAIL: $rel_path ($note)"
        continue
    fi
    if [ -s "$CASE_SNAPSHOT_STATUS" ]; then
        CASE_ACTION="exclude"
        status="FAIL"
        note=$(head -n 1 "$CASE_SNAPSHOT_STATUS")
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HAS_FAIL=1
        write_result_record "$rel_path" "$status" -1 -1 "$note" ""
        echo "  FAIL: $rel_path ($note)"
        continue
    fi

    # Every declared source is now immutable. Compile only after the full
    # closure has been copied, so an earlier compiler action cannot alter bytes
    # that a later prerequisite or extra-source snapshot would read.
    prereq_idx=0
    for prereq_src in "${prereq_sources[@]}"; do
        prereq_obj="$inc_dir/prereq_${prereq_idx}.o"
        if compile_object_with_ffc_inc "$prereq_src" "$prereq_obj" \
                "$FFC_BIN" "$inc_dir"; then
            ffc_extra+=("$prereq_obj")
        fi
        prereq_idx=$((prereq_idx + 1))
    done
    extra_idx=0
    for extra_src in "${extra_sources[@]}"; do
        extra_obj="$inc_dir/extra_${extra_idx}.o"
        if compile_object_with_extra_inc "$extra_src" "$extra_obj" \
                "$FFC_BIN" "$inc_dir"; then
            ffc_extra+=("$extra_obj")
        fi
        extra_idx=$((extra_idx + 1))
    done
    CASE_FFC_FLAGS=$(canonical_flags default "${ffc_extra[@]}")
    CASE_REF_FLAGS=$(canonical_flags '-w -J @private-module-dir' \
        "${ref_extra[@]}")

    # Step 1: compile with ffc
    if compile_with_ffc "$full_path" "$ffc_exe" "$FFC_BIN" "${ffc_extra[@]}"; then
        ffc_exit=0
        set_last_action_evidence CASE_FFC_COMPILE executed 0
    else
        ffc_exit=$?
        set_last_action_evidence CASE_FFC_COMPILE executed "$ffc_exit"
    fi

    # A lazy-mode rejection fixture is judged on the compiler exit alone: it
    # has no runnable form and no reference to compare against.
    if is_lazy_suite && lazy_negative_test "$rel_path"; then
        CASE_ACTION="reject"
        if [ "$ffc_exit" -ne 0 ]; then
            status="PASS"
            note="invalid source rejected as expected"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            status="FAIL"
            note="invalid source was accepted"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            echo "  FAIL: $rel_path (invalid source accepted)"
        fi
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        continue
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
            # Reference rejection does not turn ffc rejection into a pass.
            # There is no behavioral oracle, but feature completeness still
            # requires ffc to compile every selected, non-skipped source.
            if ! is_lazy_suite; then
                if compile_with_gfortran "$full_path" "$ref_exe" \
                        "${ref_extra[@]}"; then
                    ref_exit=0
                    set_last_action_evidence CASE_REF_COMPILE executed 0
                else
                    ref_exit=$?
                    set_last_action_evidence CASE_REF_COMPILE executed \
                        "$ref_exit"
                    if [ "$SUITE" = "fortfront-f90" ]; then
                        CASE_ACTION="exclude"
                        SKIP_COUNT=$((SKIP_COUNT + 1))
                        write_result_record "$rel_path" "SKIP" "$ffc_exit" \
                            "$ref_exit" \
                            "ffc and gfortran reject; no positive executable oracle" ""
                        echo "  SKIP: $rel_path (ffc and gfortran reject; no positive executable oracle)"
                        continue
                    fi
                    status="FAIL"
                    note="ffc compilation failed; gfortran also rejects"
                    FAIL_COUNT=$((FAIL_COUNT + 1))
                    HAS_FAIL=1
                    write_result_record "$rel_path" "$status" "$ffc_exit" \
                        "$ref_exit" "$note" "$warning_expectation"
                    echo "  FAIL: $rel_path (ffc and gfortran rejected)"
                    continue
                fi
            fi
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
    run_capture "$ffc_exe" "$ffc_out" "$TIMEOUT" ffc_run
    ffc_exit=$?
    CASE_FFC_RUN_ACTION="executed"
    CASE_FFC_RUN_EXIT=$ffc_exit
    CASE_FFC_RUN_TERMINATION=$RUN_CAPTURE_TERMINATION
    CASE_FFC_RUN_SIGNAL=$RUN_CAPTURE_SIGNAL

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
    # With --ref-cache, a previously measured reference for byte-identical
    # inputs and the same gfortran replaces the compile and the run.
    ref_cached=0
    ref_cache_entry=""
    ref_compile_status=1
    if [ -n "$REF_CACHE_DIR" ]; then
        ref_cache_key=$(reference_cache_key "$full_path" "${ref_extra[@]}")
        ref_cache_entry="$REF_CACHE_DIR/${ref_cache_key:0:2}/$ref_cache_key"
        if [ -f "$ref_cache_entry.ready" ] && [ -f "$ref_cache_entry.out" ]; then
            IFS=$'\t' read -r cache_schema ref_compile_status ref_exit \
                cached_output_sha < "$ref_cache_entry.ready"
            if [ "$cache_schema" = 2 ] && [ "$ref_compile_status" = 0 ] && \
                [ "$ref_exit" = 0 ] && \
                [ "$cached_output_sha" = \
                    "$(sha256_file_or_empty "$ref_cache_entry.out")" ]; then
                cp "$ref_cache_entry.out" "$ref_out"
                ref_cached=1
                set_inferred_action_evidence CASE_REF_COMPILE cache-hit 0
                set_inferred_action_evidence CASE_REF_RUN cache-hit 0
                REF_CACHE_HITS=$((REF_CACHE_HITS + 1))
            else
                reference_cache_discard "$ref_cache_entry"
            fi
        fi
    fi

    if [ "$ref_cached" -eq 0 ]; then
        if compile_with_gfortran "$full_path" "$ref_exe" "${ref_extra[@]}"; then
            ref_compile_status=0
            ref_exit=0
            set_last_action_evidence CASE_REF_COMPILE executed 0
        else
            ref_compile_status=$?
            ref_exit=$ref_compile_status
            set_last_action_evidence CASE_REF_COMPILE executed \
                "$ref_compile_status"
        fi
    fi
    if [ "$ref_compile_status" -ne 0 ]; then
        ref_exit=$ref_compile_status
    fi

    # Step 6: gfortran failed, but ffc already compiled and ran the file.
    if [ "$ref_exit" -ne 0 ]; then
        if [ -n "$noref_kind" ]; then
            status="FAIL"
            note="$noref_kind reference failed to compile"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            HAS_FAIL=1
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            echo "  FAIL: $rel_path ($noref_kind reference failed)"
            continue
        fi
        NOREF_COUNT=$((NOREF_COUNT + 1))
        IS_NOREF_RECORD=1
        NOREF_RECORD_REASON="reference-rejected"
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
    if [ "$ref_cached" -eq 0 ]; then
        run_capture "$ref_exe" "$ref_out" "$TIMEOUT" ref_run
        ref_exit=$?
        CASE_REF_RUN_ACTION="executed"
        CASE_REF_RUN_EXIT=$ref_exit
        CASE_REF_RUN_TERMINATION=$RUN_CAPTURE_TERMINATION
        CASE_REF_RUN_SIGNAL=$RUN_CAPTURE_SIGNAL
        if [ -n "$REF_CACHE_DIR" ]; then
            reference_cache_store "$ref_cache_entry" 0 "$ref_exit" "$ref_out"
        fi
    fi

    # A reference program that compiles but terminates abnormally cannot serve
    # as a behavioral oracle. Keep a successfully running ffc case visible as
    # NO-REF rather than comparing it against the reference compiler's failure.
    if [ "$ref_exit" -ne 0 ] && [ "$ffc_exit" -eq 0 ]; then
        NOREF_COUNT=$((NOREF_COUNT + 1))
        IS_NOREF_RECORD=1
        NOREF_RECORD_REASON="reference-runtime-failure"
        if check_xfail "$XFAIL_LOOKUP" "$rel_path"; then
            status="XPASS"
            note="listed in xfail manifest; gfortran terminates but ffc runs"
            XPASS_COUNT=$((XPASS_COUNT + 1))
            echo "  XPASS: $rel_path (gfortran terminates, ffc runs)"
        else
            status="PASS"
            note="gfortran terminates; ffc runs (NO-REF)"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        continue
    fi

    # Step 7b: a cached reference that does not match is never trusted. The
    # entry is discarded and the reference is rebuilt and rerun, so a stale or
    # nondeterministic cached output can only cost time, never change a
    # verdict; step 8b also needs the real executable to probe determinism.
    if [ "$ref_cached" -eq 1 ] && [ -z "$noref_kind" ] && \
        ! compare_outputs "$ffc_out" "$ref_out" "$ffc_exit" "$ref_exit"; then
        reference_cache_discard "$ref_cache_entry"
        ref_cached=0
        if compile_with_gfortran "$full_path" "$ref_exe" "${ref_extra[@]}"; then
            set_last_action_evidence CASE_REF_COMPILE executed 0
            run_capture "$ref_exe" "$ref_out" "$TIMEOUT" ref_run
            ref_exit=$?
            CASE_REF_RUN_ACTION="executed"
            CASE_REF_RUN_EXIT=$ref_exit
            CASE_REF_RUN_TERMINATION=$RUN_CAPTURE_TERMINATION
            CASE_REF_RUN_SIGNAL=$RUN_CAPTURE_SIGNAL
            reference_cache_store "$ref_cache_entry" 0 "$ref_exit" "$ref_out"
        else
            ref_exit=$?
            set_last_action_evidence CASE_REF_COMPILE executed "$ref_exit"
            set_inferred_action_evidence CASE_REF_RUN not-run -1
        fi
    fi

    if [ -n "$noref_kind" ]; then
        if [ "$ffc_exit" -eq 0 ] && [ "$ref_exit" -eq 0 ]; then
            status="PASS"
            note="no behavioral oracle ($noref_kind)"
            PASS_COUNT=$((PASS_COUNT + 1))
            NOREF_COUNT=$((NOREF_COUNT + 1))
            IS_NOREF_RECORD=1
            NOREF_RECORD_REASON="$noref_kind"
            write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
                "$note" "$warning_expectation"
            continue
        fi
        status="FAIL"
        note="$noref_kind execution did not terminate normally"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        HAS_FAIL=1
        write_result_record "$rel_path" "$status" "$ffc_exit" "$ref_exit" \
            "$note" "$warning_expectation"
        echo "  FAIL: $rel_path ($noref_kind execution failed)"
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
write_summary
publish_observation || exit 1
classification_status=0
classify_observation_report || classification_status=$?
[ "$classification_status" -le 1 ] || exit "$classification_status"

echo ""
echo "=== $SUITE summary ==="
echo "  PASS=$PASS_COUNT  XFAIL=$XFAIL_COUNT  XPASS=$XPASS_COUNT  FAIL=$FAIL_COUNT  NOREF=$NOREF_COUNT  SKIP=$SKIP_COUNT  WARNING_UNCHECKED=$WARNING_UNCHECKED_COUNT  TOTAL=$TOTAL_COUNT"
echo_sample_line
if [ -n "$REF_CACHE_DIR" ]; then
    echo "  Reference cache: $REF_CACHE_HITS hits in $REF_CACHE_DIR"
fi
echo "  Report: $REPORT"
echo "  Observations: $OBSERVATIONS"

# Exit nonzero if the classified view contains FAIL/XPASS/FLAKY records.
exit "$classification_status"
