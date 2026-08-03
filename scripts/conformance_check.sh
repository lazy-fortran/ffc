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
#   --repeat N   run each suite N times and fail on any case whose status is
#                not identical in every attempt (recorded FLAKY)
#   --sample N   measure a stratified random sample of about N files instead of
#                the whole corpus: each suite draws in proportion to its size,
#                so every suite keeps its own margin. Sampled runs report an
#                estimate with a confidence margin and are never dashboard
#                inputs.
#   --seed S     seed for --sample (default: 0)
#   --print-sample-plan  print the per-suite allocation for --sample and exit
#   --ref-cache DIR  cache gfortran reference outputs under DIR and reuse them
#   --require-pass-only  with --sample, draw only independently-oracled files
#                        outside XFAIL/SKIP/NOREF manifests and require every
#                        selected record to be a behavioral PASS
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
REPEAT=1
SAMPLE_TOTAL=""
SAMPLE_SEED=0
PRINT_SAMPLE_PLAN=0
REF_CACHE_DIR=""
NAMED_ARGS=()
REQUIRE_PASS_ONLY=0
STRICT_SCRATCH=""

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
        --repeat)
            if [ $# -lt 2 ]; then
                echo "ERROR: --repeat requires a count" >&2; exit 1
            fi
            REPEAT="$2"; shift 2 ;;
        --sample)
            if [ $# -lt 2 ]; then
                echo "ERROR: --sample requires a count" >&2; exit 1
            fi
            SAMPLE_TOTAL="$2"; shift 2 ;;
        --seed)
            if [ $# -lt 2 ]; then
                echo "ERROR: --seed requires an integer" >&2; exit 1
            fi
            SAMPLE_SEED="$2"; shift 2 ;;
        --print-sample-plan)
            PRINT_SAMPLE_PLAN=1; shift ;;
        --ref-cache)
            if [ $# -lt 2 ]; then
                echo "ERROR: --ref-cache requires a directory" >&2; exit 1
            fi
            REF_CACHE_DIR="$2"; shift 2 ;;
        --require-pass-only)
            REQUIRE_PASS_ONLY=1; shift ;;
        *)
            echo "ERROR: unknown option $1" >&2; exit 1 ;;
    esac
done

if [ "${#NAMED_ARGS[@]}" -gt 0 ] && [ -z "$SINGLE_SUITE" ]; then
    echo "ERROR: --file and --files-from require --suite" >&2
    exit 1
fi

if [ "$REQUIRE_PASS_ONLY" -eq 1 ]; then
    if [ -z "$SAMPLE_TOTAL" ]; then
        echo "ERROR: --require-pass-only requires --sample" >&2
        exit 1
    fi
    if [ "${#NAMED_ARGS[@]}" -gt 0 ]; then
        echo "ERROR: --require-pass-only cannot be combined with --file or --files-from" >&2
        exit 1
    fi
    # Keep strict-mode reports and logs private to this invocation. Legacy
    # invocations retain their historical TMPDIR filenames below.
    STRICT_SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/ffc_conformance_strict_XXXXXX") || {
        echo "ERROR: cannot create strict-mode scratch directory" >&2
        exit 1
    }
    trap 'rm -rf "$STRICT_SCRATCH"' EXIT
    echo "Strict pass-only scratch: $STRICT_SCRATCH"
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

# Stratified sample allocation. A single global sample drawn without regard to
# suite would let the largest suite swamp the others and leave the small ones
# with no usable margin, so the requested total is split in proportion to each
# suite's population (largest remainder, at least one file per suite).
suite_root_for() {
    case "$1" in
        fortfront-f90) echo "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/f90" ;;
        fortfront-lf)  echo "${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}/examples/lf" ;;
        lfortran)      echo "${FFC_LFORTRAN_DIR:-$CORPUS_PARENT/lfortran}/integration_tests" ;;
        gfortran-dg)   echo "${FFC_GFORTRAN_DG_DIR:-$CORPUS_PARENT/gcc/gcc/testsuite/gfortran.dg}" ;;
    esac
}

suite_population() {
    local suite="$1" root
    root=$(suite_root_for "$suite")
    [ -d "$root" ] || { echo 0; return 0; }
    if [ "$REQUIRE_PASS_ONLY" -eq 1 ]; then
        strict_candidate_list "$suite" "$STRICT_SCRATCH/${suite}.population"
        wc -l < "$STRICT_SCRATCH/${suite}.population"
    elif [ "$suite" = "fortfront-lf" ]; then
        find "$root" -maxdepth 1 \( -name '*.lf' -o -name '*.f90' \) -type f | \
            wc -l
    else
        find "$root" -maxdepth 1 -name '*.f90' -type f | wc -l
    fi
}

suite_manifest_path() {
    local suite="$1" kind="$2" safe_suite
    safe_suite=${suite//-/_}
    case "$kind" in
        xfail) echo "${FFC_XFAIL_MANIFEST:-$PROJECT_DIR/test/conformance/xfail_${safe_suite}.txt}" ;;
        skip) echo "${FFC_SKIP_MANIFEST:-$PROJECT_DIR/test/conformance/skip_${safe_suite}.txt}" ;;
        noref) echo "${FFC_NOREF_MANIFEST:-$PROJECT_DIR/test/conformance/noref_${safe_suite}.txt}" ;;
    esac
}

# Write suite-relative files that have a behavioral oracle and are not already
# expected failures/skips. This is intentionally a list filter instead of a
# change to conformance_gauntlet.sh, so its existing sampling semantics remain
# unchanged for callers that do not opt into strict mode.
strict_candidate_list() {
    local suite="$1" destination="$2" root manifest exclusion_file extension
    root=$(suite_root_for "$suite")
    exclusion_file="$STRICT_SCRATCH/${suite}.exclusions"
    : > "$exclusion_file"
    for manifest_kind in xfail skip noref; do
        manifest=$(suite_manifest_path "$suite" "$manifest_kind")
        if [ -f "$manifest" ]; then
            awk '!/^[[:space:]]*#/ && NF { print $1 }' "$manifest" >> "$exclusion_file"
        fi
    done
    LC_ALL=C sort -u "$exclusion_file" -o "$exclusion_file"
    case "$suite" in
        fortfront-lf)
            find "$root" -maxdepth 1 \( -name '*.lf' -o -name '*.f90' \) -type f | LC_ALL=C sort ;;
        *)
            case "$suite" in
                fortfront-f90|lfortran|gfortran-dg) extension='f90' ;;
            esac
            find "$root" -maxdepth 1 -name "*.$extension" -type f | LC_ALL=C sort ;;
    esac | while IFS= read -r full_path; do
        rel=${full_path#"$root"/}
        # A source without a main program cannot have a gfortran executable
        # oracle when selected as a standalone corpus case. Its module
        # prerequisites are still compiled when a runnable sibling is chosen.
        if [ "$suite" != "fortfront-lf" ] && ! grep -Eiq \
            '^[[:space:]]*(program|block[[:space:]]+data)([[:space:]]|$)' \
            "$full_path"; then
            continue
        fi
        if ! grep -Fqx -- "$rel" "$exclusion_file"; then
            printf '%s\n' "$rel"
        fi
    done > "$destination"
}

declare -A SAMPLE_FOR=()

plan_sample() {
    local suite population total=0 populations="" plan
    for suite in $SUITES; do
        population=$(suite_population "$suite")
        populations="$populations$suite $population"$'\n'
        total=$((total + population))
    done
    plan=$(printf '%s' "$populations" | awk -v want="$SAMPLE_TOTAL" \
        -v total="$total" '
        { suite[NR] = $1; population[NR] = $2; count = NR }
        END {
            if (total <= 0) exit 0
            if (want > total) want = total
            assigned = 0
            for (i = 1; i <= count; i++) {
                exact = want * population[i] / total
                alloc[i] = int(exact)
                if (alloc[i] < 1 && population[i] > 0) alloc[i] = 1
                remainder[i] = exact - int(exact)
                assigned += alloc[i]
            }
            while (assigned < want) {
                best = 0
                for (i = 1; i <= count; i++) {
                    if (alloc[i] < population[i] &&
                        (best == 0 || remainder[i] > remainder[best])) best = i
                }
                if (best == 0) break
                alloc[best]++
                remainder[best] = -1
                assigned++
            }
            for (i = 1; i <= count; i++) {
                printf "%s\t%d\t%d\n", suite[i], alloc[i], population[i]
            }
        }')
    printf '%s\n' "$plan"
}

if [ -n "$SAMPLE_TOTAL" ]; then
    case "$SAMPLE_TOTAL" in
        ''|*[!0-9]*) echo "ERROR: --sample requires a positive integer" >&2
            exit 1 ;;
    esac
    if [ "$SAMPLE_TOTAL" -lt 1 ]; then
        echo "ERROR: --sample requires a positive integer" >&2
        exit 1
    fi
    if [ "${#NAMED_ARGS[@]}" -gt 0 ]; then
        echo "ERROR: --sample cannot be combined with --file or --files-from" >&2
        exit 1
    fi
    SAMPLE_PLAN=$(plan_sample)
    echo "=== Stratified sample plan (seed $SAMPLE_SEED) ==="
    while IFS=$'\t' read -r plan_suite plan_alloc plan_population; do
        [ -z "${plan_suite:-}" ] && continue
        SAMPLE_FOR["$plan_suite"]="$plan_alloc"
        printf '  %-14s %s of %s files\n' "$plan_suite" "$plan_alloc" \
            "$plan_population"
    done <<< "$SAMPLE_PLAN"
    echo ""
elif [ "$PRINT_SAMPLE_PLAN" -eq 1 ]; then
    echo "ERROR: --print-sample-plan requires --sample" >&2
    exit 1
fi

if [ "$PRINT_SAMPLE_PLAN" -eq 1 ]; then
    exit 0
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
    if [ "$REQUIRE_PASS_ONLY" -eq 1 ]; then
        REPORT="$STRICT_SCRATCH/${SUITE}.jsonl"
        LOG="$STRICT_SCRATCH/${SUITE}.out"
    else
        REPORT="${TMPDIR:-/tmp}/ffc_conformance_${SUITE}.jsonl"
        LOG="${TMPDIR:-/tmp}/ffc_conformance_${SUITE}.out"
    fi

    echo "=== Running suite: $SUITE ==="

    rm -f "$REPORT" "$LOG"

    gauntlet_args=(--suite "$SUITE" --ffc "$FFC_BIN" \
        --report "$REPORT" --timeout "$TIMEOUT")
    gauntlet_args+=(--repeat "$REPEAT")
    if [ -n "$REF_CACHE_DIR" ]; then
        gauntlet_args+=(--ref-cache "$REF_CACHE_DIR")
    fi
    if [[ -v "SAMPLE_FOR[$SUITE]" ]] && [ "${SAMPLE_FOR[$SUITE]}" -gt 0 ]; then
        gauntlet_args+=(--sample "${SAMPLE_FOR[$SUITE]}" \
            --seed "$SAMPLE_SEED")
    fi
    if [ "$REQUIRE_PASS_ONLY" -eq 1 ]; then
        strict_candidate_list "$SUITE" "$STRICT_SCRATCH/${SUITE}.candidates"
        gauntlet_args+=(--files-from "$STRICT_SCRATCH/${SUITE}.candidates")
    fi
    gauntlet_args+=("${NAMED_ARGS[@]}")
    if bash "$GAUNTLET" "${gauntlet_args[@]}" > "$LOG" 2>&1; then
        suite_exit=0
    else
        suite_exit=$?
    fi

    # Print the log (summary line and any FAIL/XPASS)
    grep -E '(===|PASS=|FAIL:|XPASS:|ERROR:|SAMPLED:|Reference cache:)' \
        "$LOG" || true
    echo ""

    # Parse summary
    if [ -f "$REPORT" ]; then
        # A caller may deliberately reuse a scratch directory while another
        # bounded invocation is finishing.  Consume one complete summary line
        # rather than concatenating duplicate records into shell integers.
        summary=$(grep '"status":"SUMMARY"' "$REPORT" | tail -n 1 || echo "")
        if [ -n "$summary" ]; then
            fail_count=$(echo "$summary" | grep -o '"fail":[0-9][0-9]*' | grep -o '[0-9][0-9]*')
            xpass_count=$(echo "$summary" | grep -o '"xpass":[0-9][0-9]*' | grep -o '[0-9][0-9]*')
            pass_count=$(echo "$summary" | grep -o '"pass":[0-9][0-9]*' | grep -o '[0-9][0-9]*')
            xfail_count=$(echo "$summary" | grep -o '"xfail":[0-9][0-9]*' | grep -o '[0-9][0-9]*')
            noref_count=$(echo "$summary" | grep -o '"noref":[0-9][0-9]*' | grep -o '[0-9][0-9]*' || true)
            skip_count=$(echo "$summary" | grep -o '"skip":[0-9][0-9]*' | grep -o '[0-9][0-9]*' || true)
            warning_count=$(echo "$summary" | grep -o '"warning_unchecked":[0-9][0-9]*' | grep -o '[0-9][0-9]*' || true)
            total_count=$(echo "$summary" | grep -o '"total":[0-9][0-9]*' | grep -o '[0-9][0-9]*')

            echo "  $SUITE: PASS=$pass_count XFAIL=$xfail_count XPASS=$xpass_count FAIL=$fail_count TOTAL=$total_count"

            # The flaky field is present only when the count is nonzero.
            flaky_count=$(echo "$summary" | grep -o '"flaky":[0-9]*' | \
                grep -o '[0-9][0-9]*' || true)
            if [ "${fail_count:-0}" -gt 0 ]; then
                HAS_FAIL=1
            fi
            if [ "${flaky_count:-0}" -gt 0 ]; then
                HAS_FAIL=1
                echo "  $SUITE: FLAKY=$flaky_count (unstable across attempts)"
                grep '"status":"FLAKY"' "$REPORT" || true
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
            if [ "$REQUIRE_PASS_ONLY" -eq 1 ] && {
                [ "${pass_count:-0}" -ne "${total_count:-0}" ] ||
                [ "${xfail_count:-0}" -ne 0 ] ||
                [ "${xpass_count:-0}" -ne 0 ] ||
                [ "${fail_count:-0}" -ne 0 ] ||
                [ "${flaky_count:-0}" -ne 0 ] ||
                [ "${noref_count:-0}" -ne 0 ] ||
                [ "${skip_count:-0}" -ne 0 ] ||
                [ "${warning_count:-0}" -ne 0 ];
            }; then
                HAS_FAIL=1
                echo "  $SUITE: strict pass-only gate failed (every selected file must be an independently-oracled PASS)"
                if [ "${noref_count:-0}" -ne 0 ]; then
                    echo "  $SUITE: selected NOREF records:"
                    grep '"noref":true' "$REPORT" || true
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
