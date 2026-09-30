#!/usr/bin/env bash
# Freeze gate for lowering_context_t fields (Phase 3, "freeze new fields on
# lowering_context_t").
#
# lowering_context_t carries ~90 fields. Every new field is state another
# lowering path must reason about, and the whole point of the descriptor/model
# work in Phase 3 is to SHRINK it. Without a mechanical check the freeze is a
# sentence in a doc: adding a field is invisible in review because the type is
# 196 lines long and one more line looks like nothing.
#
# The allowlist is the frozen surface. A NEW field fails the gate; the change
# must either not add the field or shrink the context elsewhere and say why in
# the same commit. A REMOVED field also fails until the allowlist is shrunk in
# that same commit, so the allowlist cannot drift into recording fields that no
# longer exist (a stale allowlist would let the next addition through for free).
#
# Vacuous-pass guard: if the extractor yields too few fields or is missing any
# known anchor, the gate fails instead of comparing an empty set against itself.
# Every gate in this repo that compares "current" to "recorded" needs one, and
# this file had none before.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src/session_program_lowering_types.f90"
ALLOW="${1:-$ROOT/test/fixtures/lowering_context_fields.allow}"
TYPE_NAME="lowering_context_t"

fail=0

extract_fields() {
    # Capture EVERY name on a declaration line: Fortran allows
    # `integer :: a, b`, and an extractor that took only the first name would
    # let a new field ride in on an existing multi-name line unnoticed.
    awk "/type, public :: ${TYPE_NAME}/,/end type ${TYPE_NAME}/" "$SRC" |
        grep -E "^[[:space:]]+(integer|logical|character|real|class|type|procedure)[^:]*::" |
        sed 's/.*::[[:space:]]*//' |
        tr ',' '\n' |
        sed 's/=.*//' | tr -d ' \t' | tr 'A-Z' 'a-z' |
        sed 's/[^a-z_0-9].*//' |
        grep -E '^[a-z_][a-z_0-9]*$' |
        grep -v "^${TYPE_NAME}\$" | sort -u
}

if [ ! -f "$SRC" ]; then
    echo "FAIL: $SRC not found"
    exit 1
fi
if [ ! -f "$ALLOW" ]; then
    echo "FAIL: allowlist not found: $ALLOW"
    exit 1
fi

current="$(mktemp)"; trap 'rm -f "$current"' EXIT
extract_fields > "$current"

count=$(wc -l < "$current")

# Anchors that have been on the context for a long time; if any is missing the
# extractor is broken, not the type.
for anchor in arena session symbols symbol_count binding_table; do
    if ! grep -qxF "$anchor" "$current"; then
        echo "FAIL: extractor missed anchor field '$anchor'"
        echo "      (a broken extractor would compare nothing to nothing and pass)"
        fail=1
    fi
done
if [ "$count" -lt 50 ]; then
    echo "FAIL: extractor found only $count fields; expected the full context"
    fail=1
fi
[ "$fail" -eq 0 ] || exit 1

added=$(comm -13 "$ALLOW" "$current")
removed=$(comm -23 "$ALLOW" "$current")

if [ -n "$added" ]; then
    echo "FAIL: lowering_context_t gained $(echo "$added" | wc -l) field(s);"
    echo "      the frozen surface is test/fixtures/lowering_context_fields.allow"
    echo "${added}" | sed 's/^/  NEW FIELD: /'
    echo "  Either carry this state somewhere other than the shared context, or"
    echo "  retire an existing field in the same commit and record the swap."
    fail=1
fi

if [ -n "$removed" ]; then
    echo "FAIL: allowlist is stale - these fields no longer exist on the context:"
    echo "${removed}" | sed 's/^/  GONE: /'
    echo "  Shrink test/fixtures/lowering_context_fields.allow in the same commit"
    echo "  so the frozen surface keeps describing the real type."
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "OK: lowering_context_t frozen at $count fields, allowlist in sync"
fi
exit "$fail"
