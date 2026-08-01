#!/usr/bin/env bash
# Regenerate src/ffc_runtime_source.f90 from runtime/ffc_runtime.c.
#
# ffc links its runtime into every executable it emits, so the runtime
# source travels inside the compiler binary rather than being looked up on
# disk: a compiler and its runtime can then never be mismatched (issue #565).
# runtime/ffc_runtime.c stays the single source of truth; this script embeds
# it, and test_runtime_link_compiler fails if the two drift apart.
#
# Usage: scripts/generate_runtime_source.sh [output-path]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/runtime/ffc_runtime.c"
out="${1:-$root/src/ffc_runtime_source.f90}"

if [ ! -f "$src" ]; then
    echo "generate_runtime_source: missing $src" >&2
    exit 1
fi

long=$(awk 'length($0) > 66 { print NR": "length($0) }' "$src")
if [ -n "$long" ]; then
    echo "generate_runtime_source: runtime/ffc_runtime.c has lines over 66" >&2
    echo "columns, which do not fit the generated 88-column Fortran:" >&2
    echo "$long" >&2
    exit 1
fi

{
    cat <<'HEADER'
! GENERATED FILE - DO NOT EDIT.
!
! Regenerate with scripts/generate_runtime_source.sh after editing
! runtime/ffc_runtime.c, which is the single source of truth for the ffc
! runtime. test_runtime_link_compiler fails when this copy drifts.
!
! ffc links its runtime into every executable it emits (issue #565), so the
! runtime source ships inside the compiler binary. That makes a
! compiler/runtime version mismatch impossible by construction: there is no
! separately installed artifact to go missing or go stale.
module ffc_runtime_source
    implicit none
    private

    public :: ffc_runtime_source_text

contains

    ! The verbatim contents of runtime/ffc_runtime.c, newline-terminated.
    subroutine ffc_runtime_source_text(text)
        character(len=:), allocatable, intent(out) :: text
        character(len=1), parameter :: NL = new_line('a')

        text = ''
HEADER
    while IFS= read -r line || [ -n "$line" ]; do
        escaped=${line//\'/\'\'}
        if [ -z "$escaped" ]; then
            printf "        text = text//NL\n"
        else
            printf "        text = text// &\n            '%s'//NL\n" "$escaped"
        fi
    done < "$src"
    cat <<'FOOTER'
    end subroutine ffc_runtime_source_text

end module ffc_runtime_source
FOOTER
} > "$out"

echo "generate_runtime_source: wrote $out"
