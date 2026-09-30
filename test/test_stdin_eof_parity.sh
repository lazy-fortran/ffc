#!/usr/bin/env bash
# Unhandled end-of-file on stdin is fatal (#1351-adjacent, read(*, *) half).
#
# read(*, *) lowers to scanf. scanf reports exhaustion by returning EOF, and
# the lowering discarded that result, so an exhausted stdin left the target
# untouched and the program printed an undefined value and exited 0. Measured
# before the fix, twice, on the same binary: 32764 then 32766 - visibly
# uninitialized. gfortran reports "Fortran runtime error: End of file" for
# unit 5 and exits 2.
#
# The fix routes a runtime check through the lowering only when the READ
# carries no end=, err= AND no iostat=. Cases 3 and 4 below exist to prove
# that guard: a statement that asks to observe EOF must not be killed by the
# helper, because this stdin path emits no branch to an end= label and
# hijacking control flow would be a worse bug than the one being fixed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFC="${FFC:-$ROOT/build/fo/bin/ffc}"
WORK="$(mktemp -d /tmp/ffc-stdin-eof.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail=0
cases=0

build() {
    local tag="$1"
    cat > "$WORK/$tag.f90"
    if [ ! -s "$WORK/$tag.f90" ]; then
        echo "FAIL: fixture $tag.f90 is empty"
        fail=1
        return 1
    fi
    if ! timeout 60 "$FFC" "$WORK/$tag.f90" -o "$WORK/$tag.bin" \
            >"$WORK/$tag.build" 2>&1; then
        echo "FAIL: $tag did not compile"
        head -4 "$WORK/$tag.build"
        fail=1
        return 1
    fi
    return 0
}

# 1. Plain read(*,*) with exhausted stdin: fatal, exit 2, names end of file.
if build plain <<'EOF'
program plain
    implicit none
    integer :: i
    read (*, *) i
    print *, i
end program plain
EOF
then
    cases=$((cases + 1))
    gfortran -std=f2018 -o "$WORK/plain.g" "$WORK/plain.f90" \
        >/dev/null 2>&1
    "$WORK/plain.bin" </dev/null >"$WORK/plain.f.out" 2>&1
    rc_f=$?
    "$WORK/plain.g" </dev/null >"$WORK/plain.g.out" 2>&1
    rc_g=$?
    if [ "$rc_g" -ne 2 ]; then
        echo "FAIL: gfortran did not exit 2 on exhausted stdin (got $rc_g)"
        fail=1
    elif [ "$rc_f" -ne 2 ]; then
        echo "FAIL: over-accept, ffc exited $rc_f on exhausted stdin"
        head -2 "$WORK/plain.f.out"
        fail=1
    elif ! grep -qi "end of file" "$WORK/plain.f.out"; then
        echo "FAIL: fatal did not say end of file"
        head -3 "$WORK/plain.f.out"
        fail=1
    elif grep -q "At line 0" "$WORK/plain.f.out"; then
        echo "FAIL: diagnostic invented a line number"
        fail=1
    else
        echo "ok: exhausted stdin is fatal, exit 2, no invented line number"
    fi
fi

# 2. The same program with data on stdin still works - the guard must not
#    make ordinary reads fatal.
if [ -x "$WORK/plain.bin" ]; then
    cases=$((cases + 1))
    out=$(printf '42\n' | "$WORK/plain.bin" 2>&1)
    rc=$?
    gout=$(printf '42\n' | "$WORK/plain.g" 2>&1)
    if [ "$rc" -ne 0 ] || ! echo "$out" | grep -q "42"; then
        echo "FAIL: ordinary stdin read broke (rc=$rc out=$out)"
        fail=1
    elif [ "$(echo "$out" | tr -s ' \t' ' ')" != "$(echo "$gout" | tr -s ' \t' ' ')" ]; then
        echo "FAIL: stdin read output differs from gfortran"
        echo "  ffc=[$out] gfortran=[$gout]"
        fail=1
    else
        echo "ok: stdin read with data still works and matches gfortran"
    fi
fi

# 3. iostat= present: the statement asks to observe EOF, so the helper must
#    stay silent. Both compilers must exit 0.
cat > "$WORK/iostat_ok.f90" <<'EOF'
program iostat_ok
    implicit none
    integer :: i, ios
    read (*, *, iostat=ios) i
    print *, 'ios=', ios
end program iostat_ok
EOF
cases=$((cases + 1))
if ! gfortran -std=f2018 -fsyntax-only "$WORK/iostat_ok.f90" >/dev/null 2>&1; then
    echo "FAIL: gfortran rejects the iostat fixture"
    fail=1
elif timeout 60 "$FFC" "$WORK/iostat_ok.f90" -o "$WORK/iostat.bin" >/dev/null 2>&1; then
    "$WORK/iostat.bin" </dev/null >"$WORK/iostat.out" 2>&1
    rc=$?
    gfortran -o "$WORK/iostat.g" "$WORK/iostat_ok.f90" >/dev/null 2>&1
    "$WORK/iostat.g" </dev/null >"$WORK/iostat.g.out" 2>&1
    rcg=$?
    if [ "$rcg" -ne 0 ]; then
        echo "NOTE: gfortran exits $rcg with iostat= (adjust expectation)"
    fi
    if [ "$rc" -ne 0 ]; then
        echo "FAIL: iostat= read was killed by the EOF guard (rc=$rc)"
        head -2 "$WORK/iostat.out"
        fail=1
    else
        echo "ok: iostat= read survives exhausted stdin"
    fi
else
    echo "FAIL: iostat fixture did not compile"
    fail=1
fi

# 4. end= label present: must not be hijacked by the guard.
cat > "$WORK/end_ok.f90" <<'EOF'
program end_ok
    implicit none
    integer :: i
    read (*, *, end=99) i
    print *, i
    stop
99  print *, 'hit end'
end program end_ok
EOF
cases=$((cases + 1))
if timeout 60 "$FFC" "$WORK/end_ok.f90" -o "$WORK/end.bin" >"$WORK/end.build" 2>&1; then
    "$WORK/end.bin" </dev/null >"$WORK/end.out" 2>&1
    rc=$?
    if [ "$rc" -eq 2 ] && grep -qi "end of file" "$WORK/end.out"; then
        echo "FAIL: end= label was hijacked, fatal fired instead of branch"
        fail=1
    else
        echo "ok: end= read not hijacked (rc=$rc)"
    fi
else
    # A refusal to support end= on stdin is a capability gap, not this bug.
    if grep -qi "end=" "$WORK/end.build"; then
        echo "ok: end= on stdin refused as unsupported (not hijacked)"
    else
        echo "FAIL: end= fixture failed for an unrelated reason"
        head -3 "$WORK/end.build"
        fail=1
    fi
fi

if [ "$cases" -eq 0 ]; then
    echo "FAIL: no cases executed"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "stdin end-of-file parity: PASS ($cases cases)"
else
    echo "stdin end-of-file parity: FAILED ($cases cases)"
fi
exit "$fail"
