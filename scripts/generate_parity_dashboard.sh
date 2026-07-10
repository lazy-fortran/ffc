#!/usr/bin/env bash

set -uo pipefail
export LC_ALL=C

if [ "${BASH_VERSINFO[0]:-0}" -lt 4 ] || \
        { [ "${BASH_VERSINFO[0]:-0}" -eq 4 ] && \
          [ "${BASH_VERSINFO[1]:-0}" -lt 3 ]; }; then
    printf 'ERROR: Bash 4.3 or newer is required\n' >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib_expected_manifest.sh"
source "$SCRIPT_DIR/lib_conformance.sh"

SUITES=(fortfront-f90 fortfront-lf lfortran gfortran-dg)
declare -A REPORTS=()
MANIFEST_DIR="$PROJECT_DIR/test/conformance"
OUTPUT="$PROJECT_DIR/docs/PARITY_STATUS.md"
SNAPSHOT=""
FROM_SNAPSHOT=""
CHECK_ONLY=0

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

source "$SCRIPT_DIR/lib_parity_dashboard.sh"

is_suite() {
    case "$1" in
        fortfront-f90|fortfront-lf|lfortran|gfortran-dg) return 0 ;;
        *) return 1 ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --report)
            [ "$#" -ge 2 ] || fail "--report requires SUITE=PATH"
            assignment="$2"
            suite=${assignment%%=*}
            path=${assignment#*=}
            [ "$suite" != "$assignment" ] && [ -n "$path" ] || \
                fail "--report requires SUITE=PATH"
            is_suite "$suite" || fail "unknown suite: $suite"
            [[ -v "REPORTS[$suite]" ]] && fail "duplicate report suite: $suite"
            REPORTS["$suite"]="$path"
            shift 2
            ;;
        --manifest-dir)
            [ "$#" -ge 2 ] || fail "--manifest-dir requires a path"
            MANIFEST_DIR="$2"
            shift 2
            ;;
        --output)
            [ "$#" -ge 2 ] || fail "--output requires a path"
            OUTPUT="$2"
            shift 2
            ;;
        --snapshot)
            [ "$#" -ge 2 ] || fail "--snapshot requires a path"
            SNAPSHOT="$2"
            shift 2
            ;;
        --from-snapshot)
            [ "$#" -ge 2 ] || fail "--from-snapshot requires a path"
            FROM_SNAPSHOT="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        -h|--help)
            printf '%s\n' \
                'usage: generate_parity_dashboard.sh [--report SUITE=PATH]...' \
                '       [--manifest-dir PATH] [--output PATH] [--snapshot PATH]' \
                '       [--from-snapshot PATH] [--check]'
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[ -d "$MANIFEST_DIR" ] || fail "manifest directory not found: $MANIFEST_DIR"


CORPUS_PARENT=$(dirname "$PROJECT_DIR")
FORTFRONT_DIR=${FFC_FORTFRONT_DIR:-$CORPUS_PARENT/fortfront}
LIRIC_DIR=${FFC_LIRIC_DIR:-$CORPUS_PARENT/liric}
pin_values=$("$SCRIPT_DIR/fetch_corpora.sh" --print-pins)
pinned_lfortran=$(printf '%s\n' "$pin_values" | \
    sed -n 's/^lfortran_sha=//p')
pinned_gcc=$(printf '%s\n' "$pin_values" | sed -n 's/^gcc_sha=//p')
EXPECTED_FORTFRONT_REVISION=${FFC_DASHBOARD_FORTFRONT_REVISION:-$(git_revision "$FORTFRONT_DIR")}
EXPECTED_LIRIC_REVISION=${FFC_DASHBOARD_LIRIC_REVISION:-$(git_revision "$LIRIC_DIR")}
EXPECTED_LFORTRAN_REVISION=${FFC_DASHBOARD_LFORTRAN_REVISION:-$pinned_lfortran}
EXPECTED_GCC_REVISION=${FFC_DASHBOARD_GCC_REVISION:-$pinned_gcc}
EXPECTED_FORTFRONT_TREE=${FFC_DASHBOARD_FORTFRONT_TREE:-$(git_tree_revision "$FORTFRONT_DIR")}
EXPECTED_LIRIC_TREE=${FFC_DASHBOARD_LIRIC_TREE:-$(git_tree_revision "$LIRIC_DIR")}
EXPECTED_LFORTRAN_TREE=${FFC_DASHBOARD_LFORTRAN_TREE:-$(git_tree_revision "$(suite_root lfortran)" 2>/dev/null || printf '%040d\n' 0)}
EXPECTED_GCC_TREE=${FFC_DASHBOARD_GCC_TREE:-$(git_tree_revision "$(suite_root gfortran-dg)" 2>/dev/null || printf '%040d\n' 0)}
EXPECTED_FORTFRONT_F90_FILES=${FFC_DASHBOARD_FORTFRONT_F90_FILES:-$(suite_files_sha256 fortfront-f90)}
EXPECTED_FORTFRONT_LF_FILES=${FFC_DASHBOARD_FORTFRONT_LF_FILES:-$(suite_files_sha256 fortfront-lf)}
EXPECTED_LFORTRAN_FILES=${FFC_DASHBOARD_LFORTRAN_FILES:-$(suite_files_sha256 lfortran 2>/dev/null || printf '%064d\n' 0)}
EXPECTED_GFORTRAN_DG_FILES=${FFC_DASHBOARD_GFORTRAN_DG_FILES:-$(suite_files_sha256 gfortran-dg 2>/dev/null || printf '%064d\n' 0)}
EXPECTED_MANIFEST_SHA256=$(parity_manifest_sha256 "$MANIFEST_DIR")
EXPECTED_SCOPE_COUNT=$(parity_scope_count "$MANIFEST_DIR")


if [ -n "$FROM_SNAPSHOT" ]; then
    [ "${#REPORTS[@]}" -eq 0 ] || fail "--from-snapshot cannot be combined with --report"
    [ -s "$FROM_SNAPSHOT" ] || fail "missing snapshot: $FROM_SNAPSHOT"
    validate_snapshot_freshness "$FROM_SNAPSHOT"
    GENERATED=$(mktemp /tmp/ffc_parity_status_XXXXXX.md)
    trap 'rm -f "$GENERATED"' EXIT
    render_snapshot "$FROM_SNAPSHOT" "$GENERATED"
    if [ "$CHECK_ONLY" -eq 1 ]; then
        [ -f "$OUTPUT" ] && cmp -s "$GENERATED" "$OUTPUT" || \
            fail "dashboard is stale: $OUTPUT"
        printf 'PASS: dashboard is current: %s\n' "$OUTPUT"
    else
        mkdir -p "$(dirname "$OUTPUT")"
        cp "$GENERATED" "$OUTPUT"
        printf 'WROTE: %s\n' "$OUTPUT"
    fi
    exit 0
fi

[ -n "$SNAPSHOT" ] || SNAPSHOT="$MANIFEST_DIR/parity_dashboard.tsv"

for suite in "${SUITES[@]}"; do
    if [[ ! -v "REPORTS[$suite]" ]]; then
        REPORTS["$suite"]="/tmp/ffc_parity_${suite}.jsonl"
    fi
    [ -s "${REPORTS[$suite]}" ] || \
        fail "missing report: ${REPORTS[$suite]}"
done
TMPDIR_WORK=$(mktemp -d /tmp/ffc_parity_dashboard_XXXXXX)
trap 'rm -rf "$TMPDIR_WORK"' EXIT
ROWS="$TMPDIR_WORK/rows.tsv"
SUMMARIES="$TMPDIR_WORK/summaries.tsv"
METADATA="$TMPDIR_WORK/metadata.tsv"
SCOPES="$TMPDIR_WORK/scopes.tsv"
DISPOSITION_JOINED="$TMPDIR_WORK/disposition_joined.tsv"
JOINED="$TMPDIR_WORK/joined.tsv"
: > "$ROWS"
: > "$SUMMARIES"
: > "$METADATA"
: > "$SCOPES"

validate_report() {
    local expected_suite="$1" report="$2"
    awk -v expected_suite="$expected_suite" -v source="$report" \
        -v rows="$ROWS" -v summaries="$SUMMARIES" -v OFS=$'\t' \
        -f "$SCRIPT_DIR/validate_parity_report.awk" "$report"
}

for suite in "${SUITES[@]}"; do
    validate_report "$suite" "${REPORTS[$suite]}" || exit 1
done

ffc_revision=$(awk -F '\t' 'NR == 1 { print $10 }' "$SUMMARIES")
ffc_source_sha256=$(awk -F '\t' 'NR == 1 { print $11 }' "$SUMMARIES")
ffc_binary_sha256=$(awk -F '\t' 'NR == 1 { print $12 }' "$SUMMARIES")
fortfront_revision=$(awk -F '\t' 'NR == 1 { print $13 }' "$SUMMARIES")
fortfront_tree=$(awk -F '\t' 'NR == 1 { print $14 }' "$SUMMARIES")
liric_revision=$(awk -F '\t' 'NR == 1 { print $15 }' "$SUMMARIES")
liric_tree=$(awk -F '\t' 'NR == 1 { print $16 }' "$SUMMARIES")
git -C "$PROJECT_DIR" merge-base --is-ancestor "$ffc_revision" HEAD || \
    fail "ffc report revision is not an ancestor"
[ "$(ffc_revision_source_sha256 "$PROJECT_DIR" "$ffc_revision")" = \
    "$ffc_source_sha256" ] || fail "ffc report revision source mismatch"
[ "$fortfront_revision" = "$EXPECTED_FORTFRONT_REVISION" ] || \
    fail "stale revision: FortFront"
[ "$fortfront_tree" = "$EXPECTED_FORTFRONT_TREE" ] || \
    fail "stale tree: FortFront"
[ "$liric_revision" = "$EXPECTED_LIRIC_REVISION" ] || \
    fail "stale revision: LIRIC"
[ "$liric_tree" = "$EXPECTED_LIRIC_TREE" ] || fail "stale tree: LIRIC"
while IFS=$'\t' read -r suite _ _ _ _ _ _ _ _ ffc source_digest \
        binary_digest fortfront fortfront_content liric liric_content corpus \
        corpus_tree corpus_files; do
    [ "$ffc" = "$ffc_revision" ] || fail "stale revision: $suite ffc"
    [ "$source_digest" = "$ffc_source_sha256" ] || \
        fail "stale source digest: $suite ffc"
    [ "$binary_digest" = "$ffc_binary_sha256" ] || \
        fail "stale binary digest: $suite ffc"
    [ "$fortfront" = "$fortfront_revision" ] || \
        fail "stale revision: $suite FortFront"
    [ "$fortfront_content" = "$fortfront_tree" ] || \
        fail "stale tree: $suite FortFront"
    [ "$liric" = "$liric_revision" ] || fail "stale revision: $suite LIRIC"
    [ "$liric_content" = "$liric_tree" ] || fail "stale tree: $suite LIRIC"
    case "$suite" in
        fortfront-f90|fortfront-lf)
            [ "$corpus" = "$fortfront_revision" ] || \
                fail "stale revision: $suite corpus"
            [ "$corpus_tree" = "$fortfront_tree" ] || \
                fail "stale tree: $suite corpus"
            ;;
        lfortran)
            [ "$corpus" = "$EXPECTED_LFORTRAN_REVISION" ] || \
                fail "stale revision: LFortran corpus"
            [ "$corpus_tree" = "$EXPECTED_LFORTRAN_TREE" ] || \
                fail "stale tree: LFortran corpus"
            ;;
        gfortran-dg)
            [ "$corpus" = "$EXPECTED_GCC_REVISION" ] || \
                fail "stale revision: GCC corpus"
            [ "$corpus_tree" = "$EXPECTED_GCC_TREE" ] || \
                fail "stale tree: GCC corpus"
            ;;
    esac
    case "$suite" in
        fortfront-f90) expected_files=$EXPECTED_FORTFRONT_F90_FILES ;;
        fortfront-lf) expected_files=$EXPECTED_FORTFRONT_LF_FILES ;;
        lfortran) expected_files=$EXPECTED_LFORTRAN_FILES ;;
        gfortran-dg) expected_files=$EXPECTED_GFORTRAN_DG_FILES ;;
    esac
    [ "$corpus_files" = "$expected_files" ] || \
        fail "stale corpus file list: $suite"
done < "$SUMMARIES"

append_manifest() {
    local suite="$1" disposition="$2" path="$3"
    local lookup="$TMPDIR_WORK/lookup_${suite}_${disposition}.txt"
    validate_expected_manifest "$path" "$lookup" || exit 1
    [ -f "$path" ] || return 0
    awk -v suite="$suite" -v disposition="$disposition" '
        function trim(value) {
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            return value
        }
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        {
            line = trim($0)
            separator = index(line, " # ")
            path = trim(substr(line, 1, separator - 1))
            metadata = substr(line, separator + 3)
            split(metadata, fields, "; reason=")
            identity = fields[1]
            reason = substr(metadata, index(metadata, "; reason=") + 9)
            gsub(/\t/, " ", reason)
            print suite, path, disposition, identity, reason
        }
    ' OFS='\t' "$path" >> "$METADATA"
}

append_scope_manifest() {
    local suite="$1" path="$2"
    local lookup="$TMPDIR_WORK/lookup_${suite}_scope.txt"
    validate_expected_manifest "$path" "$lookup" || exit 1
    [ -f "$path" ] || return 0
    awk -v suite="$suite" '
        function trim(value) {
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            return value
        }
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        {
            line = trim($0)
            separator = index(line, " # ")
            path = trim(substr(line, 1, separator - 1))
            metadata = substr(line, separator + 3)
            split(metadata, fields, "; reason=")
            identity = fields[1]
            if (identity !~ /^scope=(coarray|OpenMP|OpenACC|GPU)$/) {
                printf "ERROR: invalid dashboard scope: %s\n", identity > "/dev/stderr"
                failed = 1
                next
            }
            print suite, path, identity
        }
        END { if (failed) exit 1 }
    ' OFS='\t' "$path" >> "$SCOPES"
}

for suite in "${SUITES[@]}"; do
    safe_suite=${suite//-/_}
    append_manifest "$suite" xfail "$MANIFEST_DIR/xfail_${safe_suite}.txt"
    append_manifest "$suite" skip "$MANIFEST_DIR/skip_${safe_suite}.txt"
    append_manifest "$suite" fail "$MANIFEST_DIR/fail_owners_${safe_suite}.txt"
    append_scope_manifest "$suite" "$MANIFEST_DIR/scopes_${safe_suite}.txt"
done

awk -F '\t' '
    NR == FNR {
        key = $1 SUBSEP $2
        if (key in row_status) {
            printf "ERROR: duplicate report row: %s/%s\n", $1, $2 > "/dev/stderr"
            failed = 1
        }
        row_status[key] = $3
        row_noref[key] = $4
        row_warning[key] = $5
        row_order[++row_count] = key
        next
    }
    {
        key = $1 SUBSEP $2
        if (key in metadata_disposition) {
            printf "ERROR: cross-manifest overlap: %s/%s\n", $1, $2 > "/dev/stderr"
            failed = 1
        }
        metadata_disposition[key] = $3
        metadata_identity[key] = $4
        metadata_reason[key] = $5
        metadata_order[++metadata_count] = key
    }
    END {
        for (i = 1; i <= row_count; i++) {
            key = row_order[i]
            split(key, parts, SUBSEP)
            status = row_status[key]
            expected = ""
            if (status == "XFAIL" || status == "XPASS") expected = "xfail"
            if (status == "SKIP") expected = "skip"
            if (status == "FAIL") expected = "fail"
            actual = metadata_disposition[key]
            if (expected != actual) {
                if (expected == "") {
                    printf "ERROR: unexpected metadata for %s/%s\n", parts[1], parts[2] > "/dev/stderr"
                } else {
                    printf "ERROR: missing owner for %s/%s status %s\n", parts[1], parts[2], status > "/dev/stderr"
                }
                failed = 1
                continue
            }
            identity = expected == "" ? "-" : metadata_identity[key]
            reason = expected == "" ? "-" : metadata_reason[key]
            print parts[1], parts[2], status, row_noref[key], row_warning[key], identity, reason
            used[key] = 1
        }
        for (i = 1; i <= metadata_count; i++) {
            key = metadata_order[i]
            if (!(key in used)) {
                split(key, parts, SUBSEP)
                printf "ERROR: stale metadata entry: %s/%s\n", parts[1], parts[2] > "/dev/stderr"
                failed = 1
            }
        }
        if (failed) exit 1
    }
' OFS='\t' "$ROWS" "$METADATA" > "$DISPOSITION_JOINED" || exit 1

awk -F '\t' -v scope_file="$SCOPES" '
    FILENAME == scope_file {
        key = $1 SUBSEP $2
        if (key in scope) {
            printf "ERROR: duplicate scope entry: %s/%s\n", $1, $2 > "/dev/stderr"
            failed = 1
        }
        scope[key] = $3
        next
    }
    {
        key = $1 SUBSEP $2
        row_scope = key in scope ? scope[key] : "-"
        if ($6 ~ /^scope=(coarray|OpenMP|OpenACC|GPU)$/ && $6 != row_scope) {
            printf "ERROR: disposition scope is absent from scope registry: %s/%s\n", $1, $2 > "/dev/stderr"
            failed = 1
        }
        print $0, row_scope
        used[key] = 1
    }
    END {
        for (key in scope) {
            if (!(key in used)) {
                split(key, parts, SUBSEP)
                printf "ERROR: stale scope entry: %s/%s\n", parts[1], parts[2] > "/dev/stderr"
                failed = 1
            }
        }
        if (failed) exit 1
    }
' OFS='\t' "$SCOPES" "$DISPOSITION_JOINED" > "$JOINED" || exit 1

declare -A SUBSYSTEM_FOR_OWNER=()

load_owner_subsystems() {
    local path="$MANIFEST_DIR/owner_subsystems.txt"
    local identity subsystem extra
    [ -f "$path" ] || fail "missing owner subsystem registry: $path"
    while read -r identity subsystem extra; do
        [ -z "${identity:-}" ] && continue
        [[ "$identity" == \#* ]] && continue
        [ -z "${extra:-}" ] || fail "malformed owner subsystem entry: $identity"
        [[ "$identity" == owner=* ]] || fail "invalid owner identity: $identity"
        case "$subsystem" in
            arrays|characters|control-flow|data|diagnostics|io-runtime|modules-scope|procedures|derived-oop|intrinsics|corpus|backend) ;;
            *) fail "invalid owner subsystem: $identity" ;;
        esac
        [[ ! -v "SUBSYSTEM_FOR_OWNER[$identity]" ]] || \
            fail "duplicate owner subsystem: $identity"
        SUBSYSTEM_FOR_OWNER["$identity"]="$subsystem"
    done < "$path"
    while IFS= read -r identity; do
        [[ -v "SUBSYSTEM_FOR_OWNER[$identity]" ]] || \
            fail "missing owner subsystem: $identity"
        SUBSYSTEM_FOR_OWNER["$identity"]="used:${SUBSYSTEM_FOR_OWNER[$identity]}"
    done < <(awk -F '\t' '$6 ~ /^owner=/ { print $6 }' "$JOINED" | sort -u)
    for identity in "${!SUBSYSTEM_FOR_OWNER[@]}"; do
        [[ "${SUBSYSTEM_FOR_OWNER[$identity]}" == used:* ]] || \
            fail "stale owner subsystem: $identity"
        SUBSYSTEM_FOR_OWNER["$identity"]=${SUBSYSTEM_FOR_OWNER[$identity]#used:}
    done
}

load_owner_subsystems


GENERATED_SNAPSHOT="$TMPDIR_WORK/parity_dashboard.tsv"
{
    printf 'schema\t1\n'
    printf 'revision\tffc\t%s\n' "$ffc_revision"
    printf 'revision\tFortFront\t%s\n' "$fortfront_revision"
    printf 'revision\tLIRIC\t%s\n' "$liric_revision"
    awk -F '\t' '$1 == "lfortran" { printf "revision\tLFortran\t%s\n", $17 }' \
        "$SUMMARIES"
    awk -F '\t' '$1 == "gfortran-dg" { printf "revision\tGCC\t%s\n", $17 }' \
        "$SUMMARIES"
    printf 'tree\tFortFront\t%s\n' "$fortfront_tree"
    printf 'tree\tLIRIC\t%s\n' "$liric_tree"
    printf 'tree\tLFortran\t%s\n' "$EXPECTED_LFORTRAN_TREE"
    printf 'tree\tGCC\t%s\n' "$EXPECTED_GCC_TREE"
    awk -F '\t' '{ printf "corpus-files\t%s\t%s\n", $1, $19 }' "$SUMMARIES"
    printf 'digest\tffc-source\t%s\n' "$ffc_source_sha256"
    printf 'digest\tffc-binary\t%s\n' "$ffc_binary_sha256"
    printf 'digest\tmanifests\t%s\n' "$EXPECTED_MANIFEST_SHA256"
    for suite in "${SUITES[@]}"; do emit_suite_row "$suite"; done
    emit_view_row All 0
    emit_view_row Scoped 1
    emit_excluded_row
    awk -F '\t' '
        $3 != "PASS" { counts[$1 SUBSEP $3 SUBSEP $6]++ }
        END {
            for (key in counts) {
                split(key, fields, SUBSEP)
                print fields[1], fields[2], fields[3], counts[key]
            }
        }
    ' OFS='\t' "$JOINED" | LC_ALL=C sort | \
        while IFS=$'\t' read -r suite status identity count; do
            printf 'owner\t%s\t%s\t%s\t%s\t%d\n' "$suite" "$status" \
                "$(subsystem_for_identity "$identity")" "$identity" "$count"
        done
} > "$GENERATED_SNAPSHOT"

GENERATED="$TMPDIR_WORK/PARITY_STATUS.md"
render_snapshot "$GENERATED_SNAPSHOT" "$GENERATED"

if [ "$CHECK_ONLY" -eq 1 ]; then
    [ -f "$SNAPSHOT" ] && cmp -s "$GENERATED_SNAPSHOT" "$SNAPSHOT" || \
        fail "parity snapshot is stale: $SNAPSHOT"
    [ -f "$OUTPUT" ] && cmp -s "$GENERATED" "$OUTPUT" || \
        fail "dashboard is stale: $OUTPUT"
    printf 'PASS: dashboard is current: %s\n' "$OUTPUT"
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p "$(dirname "$SNAPSHOT")"
cp "$GENERATED_SNAPSHOT" "$SNAPSHOT"
cp "$GENERATED" "$OUTPUT"
printf 'WROTE: %s\n' "$SNAPSHOT"
printf 'WROTE: %s\n' "$OUTPUT"
