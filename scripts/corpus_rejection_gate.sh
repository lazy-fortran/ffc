#!/usr/bin/env bash
# Detect newly rejected Fortran corpus files before a rejection rule merges.
#
# The report is deliberately small and expectation-neutral:
#   STATUS<TAB>project-relative-path
# A nonzero ffc compile (including timeout or signal) is REJECTED.  With
# --baseline, every baseline ACCEPTED row that becomes REJECTED is reported.
# Per-file stderr is retained in <out>.stderr.log.  Newly rejected rows are
# independently triaged with gfortran -fsyntax-only into <out>.validity.tsv;
# that oracle is evidence only and never silently permits a new rejection.
#
# Usage:
#   scripts/corpus_rejection_gate.sh --out build/rejection.tsv \
#       --corpus ../fortfront/examples --ffc build/fo/bin/ffc
#   scripts/corpus_rejection_gate.sh --out new.tsv --baseline old.tsv \
#       --allow test/conformance/rejection_allow.txt
#
# Exit: 0 clean, 1 newly rejected non-allowlisted row, 2 usage/environment.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
if [ -n "${TMPDIR-}" ]; then TMP_ROOT=$TMPDIR; else TMP_ROOT=/var/tmp/ert; fi
GFORTRAN=${FFC_GFORTRAN_ORACLE-}
[ -n "$GFORTRAN" ] || GFORTRAN=gfortran
JOBS=${FFC_REJECTION_GATE_JOBS-}
[ -n "$JOBS" ] || JOBS=3
TIMEOUT_S=${FFC_REJECTION_GATE_TIMEOUT-}
[ -n "$TIMEOUT_S" ] || TIMEOUT_S=20

OUT=
FFC=
BASELINE=
ALLOW=
CORPUS_LIST=

die() {
    echo "ERROR: $*" >&2
    exit 2
}

validate_report() {
    local report=$1
    [ -s "$report" ] || return 1
    awk -F '	' '
        NF != 2 || ($1 != "ACCEPTED" && $1 != "REJECTED") || $2 == "" {
            printf "ERROR: malformed rejection report row %d in %s\n", NR, FILENAME > "/dev/stderr"
            bad = 1
        }
        $2 in seen {
            printf "ERROR: duplicate rejection report path in %s: %s\n", FILENAME, $2 > "/dev/stderr"
            bad = 1
        }
        { seen[$2] = 1 }
        END { exit bad }
    ' "$report" || return 1
}

while (($# > 0)); do
    case "$1" in
        --out) (($# >= 2)) || die "--out requires a path"; OUT=$2; shift 2 ;;
        --corpus) (($# >= 2)) || die "--corpus requires a path"; CORPUS_LIST="$CORPUS_LIST$2
"; shift 2 ;;
        --ffc) (($# >= 2)) || die "--ffc requires a path"; FFC=$2; shift 2 ;;
        --jobs) (($# >= 2)) || die "--jobs requires a number"; JOBS=$2; shift 2 ;;
        --timeout) (($# >= 2)) || die "--timeout requires a number"; TIMEOUT_S=$2; shift 2 ;;
        --baseline) (($# >= 2)) || die "--baseline requires a path"; BASELINE=$2; shift 2 ;;
        --allow) (($# >= 2)) || die "--allow requires a path"; ALLOW=$2; shift 2 ;;
        --gfortran) (($# >= 2)) || die "--gfortran requires a path"; GFORTRAN=$2; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[ -n "$OUT" ] || die "--out is required"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ "$TIMEOUT_S" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] ||
    die "--timeout must be a positive number"

resolve_ffc() {
    local candidate newest= mtime newest_mtime=-1
    for candidate in "$PROJECT_ROOT/build/fo/bin/ffc" \
        "$PROJECT_ROOT"/build/*/app/ffc; do
        [ -x "$candidate" ] || continue
        mtime=$(stat -c '%Y' "$candidate" 2>/dev/null) || continue
        if ((mtime > newest_mtime)); then
            newest=$candidate
            newest_mtime=$mtime
        fi
    done
    printf '%s\n' "$newest"
}
[ -n "$FFC" ] || FFC=$(resolve_ffc)
[ -x "$FFC" ] || die "ffc executable not found; run 'fo build' or pass --ffc"
mkdir -p "$TMP_ROOT" || die "cannot create temporary root: $TMP_ROOT"

resolve_sibling() {
    local name=$1 candidate
    for candidate in "$PROJECT_ROOT/../code/lazy-fortran/$name" \
        "$PROJECT_ROOT/../$name" \
        "$PROJECT_ROOT/../../code/lazy-fortran/$name"; do
        if [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}
FORTFRONT_DIR=${FFC_FORTFRONT_DIR-}
[ -n "$FORTFRONT_DIR" ] || FORTFRONT_DIR=$(resolve_sibling fortfront || :)
LFORTRAN_DIR=${FFC_LFORTRAN_DIR-}
[ -n "$LFORTRAN_DIR" ] || LFORTRAN_DIR=$(resolve_sibling lfortran || :)
GCC_DIR=${FFC_GFORTRAN_DG_DIR-}
[ -n "$GCC_DIR" ] || GCC_DIR=$(resolve_sibling gcc || :)

stable_path() {
    python3 - "$1" "$PROJECT_ROOT" "$FORTFRONT_DIR" "$LFORTRAN_DIR" \
        "$GCC_DIR" <<'PY'
import os
import sys

path = os.path.realpath(sys.argv[1])
project = os.path.realpath(sys.argv[2])
roots = (
    ("fortfront", sys.argv[3]),
    ("lfortran", sys.argv[4]),
    ("gfortran-dg", sys.argv[5]),
)
for label, root in roots:
    if root:
        root = os.path.realpath(root)
        if path == root or path.startswith(root + os.sep):
            print(label + "/" + os.path.relpath(path, root))
            break
else:
    print(os.path.relpath(path, project))
PY
}

prefix_diagnostics() {
    local rel=$1 diagnostics=$2 line
    while IFS= read -r line || [ -n "$line" ]; do
        printf '%s\t%s\n' "$rel" "$line"
    done < "$diagnostics"
}

if [ -z "$CORPUS_LIST" ]; then
    [ -d "$FORTFRONT_DIR/examples" ] || die "FortFront examples not found; pass --corpus or FFC_FORTFRONT_DIR"
    CORPUS_LIST="$FORTFRONT_DIR/examples
"
    [ -d "$LFORTRAN_DIR/integration_tests" ] &&
        CORPUS_LIST="$CORPUS_LIST$LFORTRAN_DIR/integration_tests
"
    [ -d "$GCC_DIR" ] &&
        CORPUS_LIST="$CORPUS_LIST$GCC_DIR
"
fi
while IFS= read -r corpus; do
    [ -z "$corpus" ] && continue
    [ -d "$corpus" ] || die "corpus directory not found: $corpus"
done <<EOF
$CORPUS_LIST
EOF
[ -z "$BASELINE" ] || [ -f "$BASELINE" ] ||
    die "baseline report not found: $BASELINE"
[ -z "$ALLOW" ] || [ -f "$ALLOW" ] || die "allowlist not found: $ALLOW"

WORK_DIR=$(mktemp -d "$TMP_ROOT/ffc_rejection_gate_XXXXXX") ||
    die "cannot create temporary directory"
cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

FILE_LIST="$WORK_DIR/files"
FILES0="$WORK_DIR/files0"
: > "$FILES0"
while IFS= read -r corpus; do
    [ -z "$corpus" ] && continue
    find "$corpus" -type f \( -name '*.f90' -o -name '*.F90' \
        -o -name '*.f' -o -name '*.F' -o -name '*.lf' \) -print0 >> "$FILES0"
done <<EOF
$CORPUS_LIST
EOF
sort -zu -o "$FILES0" "$FILES0"
tr '\0' '\n' < "$FILES0" > "$FILE_LIST"
[ -s "$FILE_LIST" ] || die "no Fortran sources found in requested corpora"

mkdir -p "$(dirname -- "$OUT")" || die "cannot create report directory"
OUT_ABS=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT") ||
    die "cannot resolve report path"
STDERR_LOG="$OUT.stderr.log"
VALIDITY_REPORT="$OUT.validity.tsv"
VALIDITY_STDERR="$VALIDITY_REPORT.stderr.log"
: > "$WORK_DIR/file-map"
while IFS= read -r file; do
    rel=$(stable_path "$file") || die "cannot derive corpus-relative path"
    printf '%s\t%s\n' "$rel" "$file" >> "$WORK_DIR/file-map"
done < "$FILE_LIST"

probe_one() {
    file=$1
    key=$(printf '%s' "$file" | sha256sum | cut -c1-24)
    case_dir="$WORK_DIR/case_$key"
    mkdir -p "$case_dir" || return 1
    rel=$(stable_path "$file") || return 1
    timeout --kill-after="$TIMEOUT_S"s "$TIMEOUT_S"s \
        "$FFC" -c "$file" -I "$(dirname "$file")" -o "$case_dir/output.o" \
        > "$case_dir/stdout" 2> "$case_dir/stderr"
    rc=$?
    if [ "$rc" -eq 0 ]; then status=ACCEPTED; else status=REJECTED; fi
    printf '%s\t%s\n' "$status" "$rel" > "$case_dir/result"
    if [ -s "$case_dir/stderr" ]; then
        cp "$case_dir/stderr" "$case_dir/stderr.raw"
    fi
}
export -f probe_one
export -f stable_path
export FFC TIMEOUT_S WORK_DIR PROJECT_ROOT FORTFRONT_DIR LFORTRAN_DIR GCC_DIR

echo "corpus rejection gate: $(wc -l < "$FILE_LIST") files, ffc $FFC" >&2
xargs -0 -n 1 -P "$JOBS" bash -c 'probe_one "$1"' _ < "$FILES0" ||
    die "a rejection-gate worker failed"
find "$WORK_DIR" -mindepth 2 -maxdepth 2 -name result -print0 |
    xargs -0 cat | LC_ALL=C sort -k2,2 > "$WORK_DIR/report"
[ -s "$WORK_DIR/report" ] || die "workers produced no report rows"
mv "$WORK_DIR/report" "$OUT_ABS"
validate_report "$OUT_ABS" || die "invalid current rejection report"

: > "$STDERR_LOG"
while IFS= read -r result; do
    case_dir=$(dirname "$result")
    [ -f "$case_dir/stderr.raw" ] || continue
    rel=$(awk -F '\t' '{print $2}' "$result")
    prefix_diagnostics "$rel" "$case_dir/stderr.raw" >> "$STDERR_LOG"
done < <(find "$WORK_DIR" -mindepth 2 -maxdepth 2 -name result | LC_ALL=C sort)
accepted=$(grep -c '^ACCEPTED	' "$OUT_ABS" || true)
rejected=$(grep -c '^REJECTED	' "$OUT_ABS" || true)
echo "gate report: $OUT_ABS (accepted=$accepted rejected=$rejected)" >&2
[ -s "$STDERR_LOG" ] && echo "probe stderr: see $STDERR_LOG" >&2

[ -z "$BASELINE" ] && exit 0
validate_report "$BASELINE" || die "invalid baseline rejection report"
ALLOW_FILE="$WORK_DIR/allow"
: > "$ALLOW_FILE"
[ -z "$ALLOW" ] ||
    sed -e 's/[[:space:]]*$//' -e '/^[[:space:]]*#/d' \
        -e '/^[[:space:]]*$/d' "$ALLOW" > "$ALLOW_FILE"
REGRESSIONS="$WORK_DIR/regressions"
awk -F '\t' '
    NR == FNR { baseline[$2] = $1; next }
    baseline[$2] == "ACCEPTED" && $1 == "REJECTED" { print $2 }
' "$BASELINE" "$OUT_ABS" | LC_ALL=C sort -u > "$REGRESSIONS"
if [ -s "$ALLOW_FILE" ]; then
    grep -Fvx -f "$ALLOW_FILE" "$REGRESSIONS" > "$WORK_DIR/unallowed" || :
else
    cp "$REGRESSIONS" "$WORK_DIR/unallowed"
fi

: > "$VALIDITY_REPORT"
: > "$VALIDITY_STDERR"
if [ -s "$REGRESSIONS" ]; then
    while IFS= read -r rel; do
        file=$(awk -F '\t' -v wanted="$rel" '$1 == wanted {print $2; exit}' \
            "$WORK_DIR/file-map")
        if [ -z "$file" ]; then
            printf 'ERROR\t%s\n' "$rel" >> "$VALIDITY_REPORT"
            printf '%s\tmissing source for baseline row\n' "$rel" >> "$VALIDITY_STDERR"
            continue
        fi
        oracle_err="$WORK_DIR/oracle_$(printf '%s' "$rel" | sha256sum | cut -c1-16).err"
        timeout --kill-after="$TIMEOUT_S"s "$TIMEOUT_S"s \
            "$GFORTRAN" -fsyntax-only -I "$(dirname "$file")" "$file" \
            > /dev/null 2> "$oracle_err"
        oracle_rc=$?
        case "$oracle_rc" in
            0) oracle_status=VALID ;;
            124) oracle_status=TIMEOUT ;;
            126|127) oracle_status=ERROR ;;
            *) oracle_status=INVALID ;;
        esac
        printf '%s\t%s\n' "$oracle_status" "$rel" >> "$VALIDITY_REPORT"
        [ -s "$oracle_err" ] &&
            prefix_diagnostics "$rel" "$oracle_err" >> "$VALIDITY_STDERR"
    done < "$REGRESSIONS"
fi

if [ -s "$WORK_DIR/unallowed" ]; then
    count=$(wc -l < "$WORK_DIR/unallowed")
    echo "FAIL: $count file(s) accepted by the baseline are rejected now:" >&2
    cat "$WORK_DIR/unallowed" >&2
    echo "Validity triage: $VALIDITY_REPORT" >&2
    exit 1
fi
echo "OK: no newly rejected corpus files versus $BASELINE" >&2
exit 0
