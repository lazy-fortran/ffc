#!/usr/bin/env bash
# compare_conformance_reports.sh: A/B two conformance reports safely.
#
# Usage: scripts/compare_conformance_reports.sh BASELINE.jsonl CANDIDATE.jsonl
#
# ffc #642: two clean worktrees at the same commit have been observed to
# disagree on corpus results, so a delta measured across worktrees carries
# unknown error. This tool refuses such a pair outright rather than printing a
# comparison that cannot be trusted. Sound A/B measurement is same-worktree
# before/after: build and measure the baseline, apply the change, rebuild and
# measure again, in one checkout.
#
# Exit status:
#   0  reports comparable and identical per file
#   1  reports comparable, per-file statuses differ (delta printed)
#   2  reports not comparable (different worktrees, missing provenance,
#       different suites, unreadable input)

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 BASELINE.jsonl CANDIDATE.jsonl" >&2
    exit 2
fi

BASE="$1"
CAND="$2"

for path in "$BASE" "$CAND"; do
    if [ ! -f "$path" ]; then
        echo "ERROR: no such report: $path" >&2
        exit 2
    fi
done

summary_field() {
    # $1 report path, $2 field name; empty when absent.
    local summary
    summary=$(grep '"status":"SUMMARY"' "$1" | tail -n 1 || true)
    if [ -z "$summary" ]; then
        return 0
    fi
    printf '%s\n' "$summary" |
        grep -o "\"$2\":\"[^\"]*\"" |
        tail -n 1 |
        sed "s/\"$2\":\"//;s/\"$//" || true
}

base_worktree=$(summary_field "$BASE" worktree)
cand_worktree=$(summary_field "$CAND" worktree)

if [ -z "$base_worktree" ] || [ -z "$cand_worktree" ]; then
    echo "ERROR: report without worktree provenance; cannot compare." >&2
    echo "  baseline worktree:  ${base_worktree:-<missing>}" >&2
    echo "  candidate worktree: ${cand_worktree:-<missing>}" >&2
    echo "  Re-run both sides with the current runner in one worktree." >&2
    exit 2
fi

if [ "$base_worktree" != "$cand_worktree" ]; then
    echo "ERROR: reports were produced in different worktrees; refusing to" >&2
    echo "  compare (ffc #642). Measure before and after in one checkout." >&2
    echo "  baseline worktree:  $base_worktree" >&2
    echo "  candidate worktree: $cand_worktree" >&2
    exit 2
fi

base_suite=$(summary_field "$BASE" suite)
cand_suite=$(summary_field "$CAND" suite)
if [ "$base_suite" != "$cand_suite" ]; then
    echo "ERROR: reports cover different suites: $base_suite vs $cand_suite" >&2
    exit 2
fi

statuses() {
    grep -v '"status":"SUMMARY"' "$1" |
        sed -n 's/.*"file":"\([^"]*\)".*"status":"\([^"]*\)".*/\1\t\2/p' |
        sort -u
}

base_table=$(mktemp)
cand_table=$(mktemp)
trap 'rm -f "$base_table" "$cand_table"' EXIT
statuses "$BASE" > "$base_table"
statuses "$CAND" > "$cand_table"

delta=$(join -t "$(printf '\t')" -a 1 -a 2 -e '<absent>' -o '0,1.2,2.2' \
    "$base_table" "$cand_table" |
    awk -F '\t' '$2 != $3 { printf "  %s: %s -> %s\n", $1, $2, $3 }')

echo "worktree: $base_worktree"
echo "suite:    $base_suite"

if [ -z "$delta" ]; then
    echo "no per-file status changes"
    exit 0
fi

echo "per-file status changes:"
printf '%s\n' "$delta"
exit 1
