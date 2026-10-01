#!/usr/bin/env bash
# Oracle for ffc#755 (task-3d): a fixed character(len=N) function result -
# prefix-declared or allocatable-fixed - pads/truncates ON ASSIGNMENT
# (F2018 7.2.1.52) when the source is an identifier, keeping the declared
# width instead of adopting the source length. Byte-exact vs gfortran,
# RUNS>=20, md5-recorded binaries.
set -u
FFC=${FFC:-$(cd "$(dirname "$0")/.." && pwd)/build/fo/bin/ffc}
RUNS=${RUNS:-20}
TMP=$(mktemp -d /var/tmp/ffc-cfra-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
fail=0

cat > "$TMP/cfra_prefix.f90" <<'F90_A'
program main
    implicit none
    print '(a)', '['//g('hello')//']'
    print '(a)', '['//g('hi')//']'
    print '(a)', '['//g('abc')//']'
contains
    character(len=3) function g(s)
        character(len=*) :: s
        g = s
    end function g
end program main
F90_A

cat > "$TMP/cfra_prefix1.f90" <<'F90_B'
program main
    implicit none
    integer :: i
    character(len=3) :: w
    w = 'abc'
    do i = 1, 3
        print '(a,i0)', '['//two(w(i:i))//']', i
    end do
contains
    character(1) function two(t)
        character(1) :: t
        two = t
    end function two
end program main
F90_B

cat > "$TMP/cfra_repeat.f90" <<'F90_C'
program main
    implicit none
    character(len=4), allocatable :: t
    character(len=6) :: src
    src = 'abcdef'
    t = src
    print '(a)', '['//t//']'
    src = 'hi'
    t = src
    print '(a)', '['//t//']'
    t = src(1:3)
    print '(a)', '['//t//']'
end program main
F90_C

for name in cfra_prefix cfra_prefix1 cfra_repeat; do
    gfortran -std=f2018 -o "$TMP/$name.gfortran" "$TMP/$name.f90" || { echo "FAIL: gfortran build $name"; fail=1; continue; }
    "$FFC" "$TMP/$name.f90" -o "$TMP/$name.ffc" || { echo "FAIL: ffc refused $name"; fail=1; continue; }
    want=$("$TMP/$name.gfortran" | tr '\n' '|')
    md5=$(md5sum "$TMP/$name.ffc" | awk '{print $1}')
    runs=0
    while [ "$runs" -lt "$RUNS" ]; do
        got=$("$TMP/$name.ffc" 2>&1 | tr '\n' '|') || { echo "FAIL: $name crashed"; fail=1; break; }
        if [ "$got" != "$want" ]; then
            echo "FAIL: $name run $((runs+1)) got [$got] want [$want]"; fail=1; break
        fi
        runs=$((runs+1))
    done
    [ "$runs" -eq "$RUNS" ] && echo "ok: $name matches gfortran on $runs/$runs runs [$want]"
    echo "    binary md5: $md5"
done

[ "$fail" -ne 0 ] && { echo "FAIL: fixed character result assignment parity"; exit 1; }
echo "PASS: fixed character result assignment parity, $RUNS runs per shape"
