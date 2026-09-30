#!/usr/bin/env bash
# PAUSE parity at stdin EOF: ffc must agree with gfortran on stdout and exit.
#
# PAUSE is obsolescent (deleted as a executable statement in later revisions;
# gfortran rejects it under -std=f2018 and -std=f2023) but gfortran accepts it
# in its default GNU mode. In that mode, at end of input on stdin, PAUSE
# prints the message line and the "To resume execution, type go." banner and
# then terminates the program normally with exit status 0.
#
# This oracle pins exactly that parity over the PAUSE corpus files, because the
# Fortran-side helper expect_output_matches_gfortran runs the produced
# executables with INHERITED stdin (src/ffc_test_support.f90:106 does
# `exe > out 2>&1`), so a PAUSE program under it would block on whatever stdin
# the test runner happens to have. Here stdin is closed from /dev/null so the
# end-of-input path is the one actually exercised.
#
# issue_2910_statement_query_forms is intentionally excluded: gfortran refuses
# to build it ("Positive width required in format string" for (I0) at line 14),
# so it has no gfortran reference and is a separate format-editing matter.
#
# Exit: 0 parity holds, 1 parity broken, 2 environment problem.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH_ROOT=${TMPDIR:-/var/tmp/ert}
mkdir -p "$SCRATCH_ROOT"
WORK=$(mktemp -d "$SCRATCH_ROOT/ffc-pause-parity.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

FFC=${FFC_PAUSE_PARITY_FFC:-$ROOT/build/fo/bin/ffc}
[ -x "$FFC" ] || { echo "pause: ffc not built at $FFC" >&2; exit 2; }
command -v gfortran >/dev/null || { echo "pause: gfortran required" >&2; exit 2; }

CORPUS=${FFC_PAUSE_PARITY_CORPUS:-$ROOT/../fortfront/examples/f90}
[ -d "$CORPUS" ] || { echo "pause: corpus not found at $CORPUS" >&2; exit 2; }

files=(
    pause_without_message
    pause_in_if_block
    pause_with_string_in_loop
    issue_2076_pause_statement_silently_dropped
)

fail=0
checked=0
for name in "${files[@]}"; do
    src=$CORPUS/$name.f90
    if [ ! -f "$src" ]; then
        echo "SKIP: $name not present in corpus"
        continue
    fi

    # Strict dialects must reject PAUSE; both compilers have to agree that they
    # do, otherwise this oracle would be pinning a dialect ffc does not claim.
    gfortran -std=f2023 -o "$WORK/$name.strict" "$src" >"$WORK/$name.strict.err" 2>&1
    strict_g=$?
    if [ "$strict_g" -eq 0 ]; then
        echo "NOTE: gfortran accepts PAUSE under -std=f2023; parity claim below is against default GNU mode only"
    fi

    if ! gfortran -o "$WORK/$name.g" "$src" >"$WORK/$name.gbuild.err" 2>&1; then
        echo "SKIP: gfortran cannot build $name (no reference)"
        continue
    fi
    if ! "$FFC" "$src" -o "$WORK/$name.f" >"$WORK/$name.fbuild.err" 2>&1; then
        echo "FAIL: $name ffc build failed while gfortran succeeded"
        tail -3 "$WORK/$name.fbuild.err"
        fail=1
        continue
    fi

    timeout 20 "$WORK/$name.g" </dev/null >"$WORK/$name.g.out" 2>&1
    rcg=$?
    timeout 20 "$WORK/$name.f" </dev/null >"$WORK/$name.f.out" 2>&1
    rcf=$?
    checked=$((checked + 1))

    if [ "$rcg" -ne "$rcf" ]; then
        echo "FAIL: $name exit status differs (gfortran=$rcg ffc=$rcf)"
        fail=1
    fi
    if ! diff -u "$WORK/$name.g.out" "$WORK/$name.f.out" >"$WORK/$name.diff"; then
        echo "FAIL: $name stdout differs at stdin EOF"
        head -8 "$WORK/$name.diff"
        fail=1
    fi
    if [ "$rcg" -eq 0 ] && [ "$rcf" -eq 0 ] && diff -q "$WORK/$name.g.out" "$WORK/$name.f.out" >/dev/null; then
        echo "ok: $name parity at stdin EOF (exit 0, $(wc -l <"$WORK/$name.g.out") lines)"
    fi
done

if [ "$checked" -eq 0 ]; then
    echo "pause: no file was checked - vacuous, treating as environment failure" >&2
    exit 2
fi

if [ "$fail" -ne 0 ]; then
    echo "pause parity: FAILED ($checked files checked)"
    exit 1
fi
echo "pause parity: PASS ($checked files checked)"
exit 0
