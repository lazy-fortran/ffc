#!/usr/bin/env bash
# Gate: an ffc executable must not carry DT_TEXTREL.
#
# Every ffc executable used to link with DT_TEXTREL because the DIRECT/session
# path materialized globals (the synthesized E/EN helper format strings
# `.ffc.een.norm`, `.ffc.een.field`, and libc entries reached from the
# helper) with `movabs` + R_X86_64_64 inside `.text`. A 64-bit absolute
# relocation in a read-only section makes ld emit DT_TEXTREL and map text
# writable+relocated at load time.
#
# This drives the real path that had the defect: the ffc CLI compiling through
# the direct LIRIC session (ctx->jit set), which is the branch fixed in liric
# 63672bf. It is NOT covered by liric's own objfile test, which compiles with
# cc->jit == NULL and therefore passed identically before the fix.
#
# Exit: 0 clean, 1 contract violated, 2 environment problem.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH_ROOT=${TMPDIR:-/var/tmp/ert}
mkdir -p "$SCRATCH_ROOT"
WORK=$(mktemp -d "$SCRATCH_ROOT/ffc-textrel-gate.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

FFC=${FFC_TEXTREL_GATE_FFC:-$ROOT/build/fo/bin/ffc}
[ -x "$FFC" ] || { echo "gate: ffc not built at $FFC" >&2; exit 2; }
command -v readelf >/dev/null || { echo "gate: readelf required" >&2; exit 2; }

SRC=$WORK/textrel.f90
printf 'program p\n' > "$SRC"
printf '  print "(E10.3)", 1.0\n' >> "$SRC"
printf 'end program p\n' >> "$SRC"

fail=0

# Object: no absolute 64-bit relocation may reference .text.
if ! "$FFC" -c "$SRC" -o "$WORK/textrel.o" >"$WORK/c.log" 2>&1; then
    echo "gate: ffc -c failed" >&2; cat "$WORK/c.log" >&2; exit 2
fi
# Match the type token anywhere in the row: readelf -W prints
# Offset, Info, Type, SymValue+Addend, Symbol, and column order is not stable
# across binutils versions. Anchoring on a fixed column silently under-counted.
abs64=$(readelf -rW "$WORK/textrel.o" | grep -c 'R_X86_64_64' || true)
if [ "$abs64" -ne 0 ]; then
    echo "FAIL: object carries $abs64 absolute R_X86_64_64 relocation(s)"
    readelf -rW "$WORK/textrel.o" | awk '$2 ~ /R_X86_64_64$/' | head -5
    fail=1
else
    echo "ok: object has no R_X86_64_64"
fi

# The format-string globals must be reached through the GOT.
if ! readelf -rW "$WORK/textrel.o" | grep -q '\.ffc\.een\.norm'; then
    echo "FAIL: fixture lost its .ffc.een.norm reference; test no longer covers the defect"
    fail=1
elif readelf -rW "$WORK/textrel.o" | grep '\.ffc\.een\.norm' | grep -vq GOTPCREL; then
    echo "FAIL: .ffc.een.norm is not reached through the GOT"
    readelf -rW "$WORK/textrel.o" | grep '\.ffc\.een\.norm'
    fail=1
else
    echo "ok: .ffc.een.norm reached through the GOT"
fi

# Linked executable: ld must stay quiet and the ELF must not set TEXTREL.
if ! link_out=$("$FFC" "$SRC" -o "$WORK/textrel" 2>&1); then
    echo "gate: ffc link failed" >&2; echo "$link_out" >&2; exit 2
fi
if echo "$link_out" | grep -qi 'TEXTREL'; then
    echo "FAIL: linker reported a TEXTREL warning"
    echo "$link_out"
    fail=1
fi
if readelf -d "$WORK/textrel" | grep -q 'TEXTREL'; then
    echo "FAIL: executable carries DT_TEXTREL"
    readelf -d "$WORK/textrel" | grep -i textrel
    fail=1
else
    echo "ok: executable has no DT_TEXTREL"
fi

# Behaviour must not change on the way there.
if [ ! -x "$WORK/textrel" ]; then
    echo "FAIL: no executable produced"
    fail=1
else
    got=$("$WORK/textrel")
    if [ "$got" != " 0.100E+01" ]; then
        echo "FAIL: output changed: got [$got] want [ 0.100E+01]"
        fail=1
    else
        echo "ok: prints 0.100E+01"
    fi
fi

if [ "$fail" -ne 0 ]; then
    echo "textrel gate: FAILED"
    exit 1
fi
echo "textrel gate: PASS"
exit 0
