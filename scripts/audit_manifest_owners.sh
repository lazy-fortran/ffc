#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib_expected_manifest.sh"

if [ "$#" -gt 0 ]; then
    MANIFESTS=("$@")
else
    MANIFESTS=("$PROJECT_DIR"/test/conformance/xfail_*.txt)
    MANIFESTS+=("$PROJECT_DIR"/test/conformance/skip_*.txt)
fi

TMPDIR_WORK=$(mktemp -d /tmp/ffc_manifest_audit_XXXXXX)
trap 'rm -rf "$TMPDIR_WORK"' EXIT
OWNER_LIST="$TMPDIR_WORK/owners.txt"
: > "$OWNER_LIST"

for manifest in "${MANIFESTS[@]}"; do
    if [ ! -f "$manifest" ]; then
        printf 'ERROR: manifest not found: %s\n' "$manifest" >&2
        exit 1
    fi
    validate_expected_manifest "$manifest" "$TMPDIR_WORK/lookup.txt" || exit 1
    manifest_owner_references "$manifest" >> "$OWNER_LIST"
done

sort -u "$OWNER_LIST" -o "$OWNER_LIST"
failed=0
while IFS= read -r owner; do
    [ -z "$owner" ] && continue
    repository=${owner%#*}
    issue_number=${owner##*#}
    state=$(gh issue view "$issue_number" --repo "$repository" \
        --json state --jq .state 2>/dev/null) || state="MISSING"
    if [ "$state" != "OPEN" ]; then
        printf 'ERROR: owner %s is %s\n' "$owner" "$state" >&2
        failed=1
    fi
done < "$OWNER_LIST"

if [ "$failed" -ne 0 ]; then
    exit 1
fi
printf 'PASS: %s open manifest owners\n' "$(wc -l < "$OWNER_LIST")"
