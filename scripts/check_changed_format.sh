#!/usr/bin/env bash
set -euo pipefail

# Compare only files changed by this revision. Existing formatter debt is
# reported but does not turn every feature PR red; new files and regressions
# against the base revision remain blocking.
if [[ -n "${BASE_REF:-}" ]]; then
    base_ref="$BASE_REF"
elif [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    base_ref="origin/$GITHUB_BASE_REF"
else
    base_ref="HEAD^"
fi

git rev-parse --verify "$base_ref^{commit}" >/dev/null

base_file=$(mktemp --suffix=.f90)
head_file=$(mktemp --suffix=.f90)
trap 'rm -f "$base_file" "$head_file"' EXIT

failed=0
while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    case "$path" in
        *.f|*.f90|*.f95|*.f03|*.f08|*.f15|*.F|*.F90|*.F95|*.F03|*.F08|*.F15) ;;
        *) continue ;;
    esac

    cp -- "$path" "$head_file"
    head_rc=0
    fo fmt --check "$head_file" >/dev/null 2>&1 || head_rc=$?

    if git cat-file -e "$base_ref:$path" 2>/dev/null; then
        git show "$base_ref:$path" > "$base_file"
        base_rc=0
        fo fmt --check "$base_file" >/dev/null 2>&1 || base_rc=$?
    else
        base_rc=2
    fi

    if [[ "$base_rc" -eq 0 && "$head_rc" -ne 0 ]]; then
        echo "format regression: $path"
        failed=1
    elif [[ "$base_rc" -ne 0 && "$head_rc" -ne 0 ]]; then
        echo "format debt (pre-existing): $path"
    elif [[ "$base_rc" -ne 0 && "$head_rc" -eq 0 ]]; then
        echo "format repaired: $path"
    else
        echo "format ok: $path"
    fi
done < <(git diff --name-only --diff-filter=ACMR "$base_ref...HEAD")

if [[ "$failed" -ne 0 ]]; then
    echo "changed-file formatting regressed relative to $base_ref" >&2
    exit 1
fi
