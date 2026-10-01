#!/usr/bin/env bash
# Oracle for LOGICAL array-section mask reductions (task-3e/#339 prerequisite
# that previously blocked the cluster): count/any/all over a(lo:hi[:st]),
# including stride and reverse sections, single elements, if() use, and
# whole-array forms, byte-exact vs gfortran.
set -u
FFC=${FFC:-$(cd "$(dirname "$0")/.." && pwd)/build/fo/bin/ffc}
RUNS=${RUNS:-20}
TMP=$(mktemp -d /var/tmp/ffc-lsec-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
fail=0

cat > "$TMP/mask_section_basic.f90" <<'F90_A'
program main
    implicit none
    logical :: l(5)
    integer :: i
    l(1) = .true.
    l(2) = .false.
    l(3) = .true.
    l(4) = .true.
    l(5) = .false.
    do i = 2, 4
        if (l(i)) print *, 'T'
    end do
    print *, count(l(2:4)), count(l)
    print *, any(l(1:2)), all(l(4:5))
    print *, any(l(2:2)), count(l(5:5))
end program main
F90_A

cat > "$TMP/mask_section_stride.f90" <<'F90_B'
program main
    implicit none
    logical :: l(7)
    integer :: i
    do i = 1, 7
        l(i) = mod(i, 3) /= 0
    end do
    print *, count(l(1:7:2)), any(l(2:7:3)), all(l(1:6:2))
    print *, count(l(:)), count(l(7:1:-2))
end program main
F90_B

cat > "$TMP/mask_section_gate.f90" <<'F90_C'
program main
    implicit none
    logical :: m(4)
    m(1) = .false.
    m(2) = .true.
    m(3) = .false.
    m(4) = .true.
    if (any(m(1:3))) then
        print *, 'ANY'
    else
        print *, 'NONE'
    end if
    if (all(m(2:4))) then
        print *, 'ALL4'
    else
        print *, 'NOTALL'
    end if
    print *, count(m) - count(m(2:3))
end program main
F90_C

for name in mask_section_basic mask_section_stride mask_section_gate; do
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

[ "$fail" -ne 0 ] && { echo "FAIL: logical section mask reduction parity"; exit 1; }
echo "PASS: logical section mask reduction parity, $RUNS runs per shape"
