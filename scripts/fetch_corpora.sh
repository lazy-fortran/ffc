#!/usr/bin/env bash
# fetch_corpora.sh: fetch external conformance corpora next to this checkout.
#
# Usage:
#   scripts/fetch_corpora.sh [--verify-only] [lfortran|gfortran-dg ...]
#   scripts/fetch_corpora.sh --print-pins
#
# With no corpus arguments, fetches or verifies both pinned corpora.
#
# Destinations match the conformance gauntlet defaults (sibling directories
# under the parent of this repository) and honor the same overrides:
#   FFC_LFORTRAN_DIR    default: ../lfortran
#   FFC_GFORTRAN_DG_DIR default: ../gcc/gcc/testsuite/gfortran.dg
#
# The gcc checkout is a blobless sparse checkout of gcc/testsuite/gfortran.dg
# only. No foreign source files are copied into this repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib_conformance.sh"
REPO_ROOT="$(resolve_primary_checkout_root "$(dirname "$SCRIPT_DIR")")"
PARENT="$(dirname "$REPO_ROOT")"

LFORTRAN_URL="https://github.com/lfortran/lfortran"
GCC_URL="https://github.com/gcc-mirror/gcc"
LFORTRAN_SHA="caf87b660f803148f000046392a5da803f9fc630"
GCC_SHA="395e3d8131c189cd58e8c8061cdc77d1c44e3822"

if [ "${FFC_CORPUS_TEST_MODE:-0}" = "1" ]; then
    LFORTRAN_URL="${FFC_LFORTRAN_URL:?}"
    GCC_URL="${FFC_GCC_URL:?}"
    LFORTRAN_SHA="${FFC_LFORTRAN_SHA:?}"
    GCC_SHA="${FFC_GCC_SHA:?}"
fi

VERIFY_ONLY=0
CORPORA=()

for pin in "$LFORTRAN_SHA" "$GCC_SHA"; do
    if [[ ! "$pin" =~ ^[0-9a-f]{40}$ ]]; then
        echo "ERROR: corpus pin must be a lowercase 40-hex SHA: $pin" >&2
        exit 2
    fi
done

for arg in "$@"; do
    case "$arg" in
        --verify-only) VERIFY_ONLY=1 ;;
        --print-pins)
            printf 'lfortran_sha=%s\ngcc_sha=%s\n' "$LFORTRAN_SHA" "$GCC_SHA"
            exit 0 ;;
        lfortran|gfortran-dg) CORPORA+=("$arg") ;;
        *) echo "usage: $0 [--verify-only] [lfortran|gfortran-dg ...]" >&2; exit 2 ;;
    esac
done
[ ${#CORPORA[@]} -eq 0 ] && CORPORA=(lfortran gfortran-dg)

normalized_url() {
    local url="$1"
    url=${url%/}
    printf '%s\n' "${url%.git}"
}

verify_checkout() {
    local name="$1" dir="$2" url="$3" sha="$4" required_dir="$5"
    local actual_url actual_sha
    if [ ! -d "$dir/.git" ]; then
        echo "ERROR: $name checkout missing at $dir" >&2
        return 1
    fi
    actual_url=$(git -C "$dir" config --get remote.origin.url)
    if [ "$(normalized_url "$actual_url")" != "$(normalized_url "$url")" ]; then
        echo "ERROR: $name origin is $actual_url, expected $url" >&2
        return 1
    fi
    if [ -n "$(git -C "$dir" status --porcelain)" ]; then
        echo "ERROR: $name checkout is dirty at $dir" >&2
        return 1
    fi
    actual_sha=$(git -C "$dir" rev-parse HEAD)
    if [ "$actual_sha" != "$sha" ]; then
        echo "ERROR: $name HEAD is $actual_sha, expected $sha" >&2
        return 1
    fi
    if git -C "$dir" symbolic-ref -q HEAD >/dev/null; then
        echo "ERROR: $name checkout must be detached at $sha" >&2
        return 1
    fi
    if [ ! -d "$required_dir" ] || \
        ! find "$required_dir" -maxdepth 1 -name '*.f90' -type f -print -quit | grep -q .; then
        echo "ERROR: $name corpus is incomplete at $required_dir" >&2
        return 1
    fi
    echo "$name pinned at $actual_sha"
}

fetch_exact() {
    local name="$1" dir="$2" url="$3" sha="$4" required_dir="$5"
    local sparse_path="${6:-}"
    local object_filter="${7:-}"
    local fetch_args=(--depth 1)
    if [ -d "$dir/.git" ]; then
        verify_checkout "$name" "$dir" "$url" "$sha" "$required_dir"
        return
    fi
    if [ "$VERIFY_ONLY" -eq 1 ]; then
        echo "ERROR: $name checkout missing at $dir" >&2
        return 1
    fi
    if [ -e "$dir" ] && \
        find "$dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        echo "ERROR: $name path exists but is not a git checkout: $dir" >&2
        return 1
    fi
    mkdir -p "$dir"
    git -C "$dir" init
    git -C "$dir" remote add origin "$url"
    if [ -n "$sparse_path" ]; then
        git -C "$dir" sparse-checkout init --cone
        git -C "$dir" sparse-checkout set "$sparse_path"
    fi
    if [ -n "$object_filter" ]; then
        fetch_args+=("--filter=$object_filter")
    fi
    git -C "$dir" fetch "${fetch_args[@]}" origin "$sha"
    git -C "$dir" checkout --detach "$sha"
    verify_checkout "$name" "$dir" "$url" "$sha" "$required_dir"
}

fetch_lfortran() {
    local dir="${FFC_LFORTRAN_DIR:-$PARENT/lfortran}"
    fetch_exact "lfortran" "$dir" "$LFORTRAN_URL" "$LFORTRAN_SHA" \
        "$dir/integration_tests"
}

fetch_gfortran_dg() {
    local dg_dir="${FFC_GFORTRAN_DG_DIR:-$PARENT/gcc/gcc/testsuite/gfortran.dg}"
    local gcc_root="${dg_dir%/gcc/testsuite/gfortran.dg}"
    if [ "$gcc_root" = "$dg_dir" ]; then
        echo "ERROR: FFC_GFORTRAN_DG_DIR must end in gcc/testsuite/gfortran.dg" >&2
        return 2
    fi
    fetch_exact "gcc" "$gcc_root" "$GCC_URL" "$GCC_SHA" "$dg_dir" \
        "gcc/testsuite/gfortran.dg" "blob:none"
    if [ "$(git -C "$gcc_root" sparse-checkout list)" != \
        "gcc/testsuite/gfortran.dg" ]; then
        echo "ERROR: gcc sparse checkout does not select gfortran.dg" >&2
        return 1
    fi
    echo "gfortran.dg files: $(find "$dg_dir" -maxdepth 1 -name '*.f90' | wc -l)"
}

for corpus in "${CORPORA[@]}"; do
    case "$corpus" in
        lfortran) fetch_lfortran ;;
        gfortran-dg) fetch_gfortran_dg ;;
    esac
done
