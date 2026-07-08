#!/usr/bin/env bash
# fetch_corpora.sh: fetch external conformance corpora next to this checkout.
#
# Usage:
#   scripts/fetch_corpora.sh [--update] [lfortran|gfortran-dg ...]
#
# With no corpus arguments, fetches all corpora. Existing checkouts are
# left untouched unless --update is given, which pulls the latest upstream.
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

UPDATE=0
CORPORA=()

for arg in "$@"; do
    case "$arg" in
        --update) UPDATE=1 ;;
        lfortran|gfortran-dg) CORPORA+=("$arg") ;;
        *) echo "usage: $0 [--update] [lfortran|gfortran-dg ...]" >&2; exit 2 ;;
    esac
done
[ ${#CORPORA[@]} -eq 0 ] && CORPORA=(lfortran gfortran-dg)

fetch_lfortran() {
    local dir="${FFC_LFORTRAN_DIR:-$PARENT/lfortran}"
    if [ -d "$dir/.git" ]; then
        if [ "$UPDATE" -eq 1 ]; then
            echo "updating lfortran at $dir"
            git -C "$dir" pull --ff-only
        else
            echo "lfortran present at $dir ($(git -C "$dir" rev-parse --short HEAD))"
        fi
    else
        echo "cloning lfortran to $dir"
        git clone --depth 1 "$LFORTRAN_URL" "$dir"
    fi
}

fetch_gfortran_dg() {
    local dg_dir="${FFC_GFORTRAN_DG_DIR:-$PARENT/gcc/gcc/testsuite/gfortran.dg}"
    local gcc_root="${dg_dir%/gcc/testsuite/gfortran.dg}"
    if [ "$gcc_root" = "$dg_dir" ]; then
        echo "FFC_GFORTRAN_DG_DIR must end in gcc/testsuite/gfortran.dg" >&2
        exit 2
    fi
    if [ -d "$gcc_root/.git" ]; then
        if [ "$UPDATE" -eq 1 ]; then
            echo "updating gcc sparse checkout at $gcc_root"
            git -C "$gcc_root" pull --ff-only
        else
            echo "gcc present at $gcc_root ($(git -C "$gcc_root" rev-parse --short HEAD))"
        fi
    else
        echo "sparse-cloning gcc gfortran.dg to $gcc_root"
        git clone --filter=blob:none --no-checkout --depth 1 "$GCC_URL" "$gcc_root"
        git -C "$gcc_root" sparse-checkout set gcc/testsuite/gfortran.dg
        git -C "$gcc_root" checkout
    fi
    echo "gfortran.dg files: $(find "$dg_dir" -maxdepth 1 -name '*.f90' | wc -l)"
}

for corpus in "${CORPORA[@]}"; do
    case "$corpus" in
        lfortran) fetch_lfortran ;;
        gfortran-dg) fetch_gfortran_dg ;;
    esac
done
