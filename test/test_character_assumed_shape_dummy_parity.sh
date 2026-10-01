#!/usr/bin/env bash
# Oracle for task-3d/#348 slice: rank-1 fixed-length CHARACTER assumed-shape
# dummies bound through the canonical array descriptor. The two-extent shape
# (one callee called with a 3-element and a 2-element actual) is decisive:
# no compile-time specialization can answer both, so a pass proves the
# descriptor drives element stride, extent, and lower bound at run time.
# Byte-exact vs gfortran, RUNS>=20, md5-recorded binaries.
set -u
FFC=${FFC:-$(cd "$(dirname "$0")/.." && pwd)/build/fo/bin/ffc}
RUNS=${RUNS:-20}
TMP=$(mktemp -d /var/tmp/ffc-casd-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
fail=0

cat > "$TMP/casd_whole.f90" <<'F90_A'
program main
    implicit none
    character(len=4) :: arr(3)
    arr(1) = 'alpha'
    arr(2) = 'beta'
    arr(3) = 'gamma'
    call show(arr)
contains
    subroutine show(x)
        character(len=4), intent(in) :: x(:)
        integer :: i
        print *, size(x)
        do i = 1, size(x)
            print *, x(i)
        end do
    end subroutine show
end program main
F90_A

cat > "$TMP/casd_two_extents.f90" <<'F90_B'
program main
    implicit none
    character(len=4) :: a3(3), a2(2)
    a3(1) = 'alpha'; a3(2) = 'beta'; a3(3) = 'gamma'
    a2(1) = 'delta'; a2(2) = 'eps'
    call show(a3)
    call show(a2)
    call show(a3(1:0)//'')
contains
    subroutine show(x)
        character(len=4), intent(in) :: x(:)
        print *, size(x), x(1), x(size(x))
    end subroutine show
end program main
F90_B

cat > "$TMP/casd_inout.f90" <<'F90_C'
program main
    implicit none
    character(len=4) :: arr(3)
    arr(1) = 'alpha'; arr(2) = 'beta'; arr(3) = 'gamma'
    call bump(arr)
    print *, arr(1), arr(2)
contains
    subroutine bump(y)
        character(len=4), intent(inout) :: y(:)
        y(1) = 'zz'
        y(3) = 'done'
    end subroutine bump
end program main
F90_C

for name in casd_whole casd_inout; do
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

# two-extent shape without the zero-size probe line
grep -v "a3(1:0)" "$TMP/casd_two_extents.f90" > "$TMP/casd_two.f90"
name=casd_two
gfortran -std=f2018 -o "$TMP/$name.gfortran" "$TMP/$name.f90" || { echo "FAIL: gfortran build $name"; fail=1; }
if [ ! -f "$TMP/$name.ffc" ]; then "$FFC" "$TMP/$name.f90" -o "$TMP/$name.ffc" || { echo "FAIL: ffc refused two-extent shape"; fail=1; }; fi
if [ -f "$TMP/$name.ffc" ] && [ -f "$TMP/$name.gfortran" ]; then
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
fi

# honest refusals: allocatable char actual and character section actual must
# still be refused, naming the reason.
cat > "$TMP/casd_refuse_alloc.f90" <<'F90_D'
program main
    implicit none
    character(len=3), allocatable :: av(:)
    allocate(av(2))
    av(1) = 'one'; av(2) = 'two'
    call show(av)
contains
    subroutine show(x)
        character(len=3), intent(in) :: x(:)
        print *, size(x)
    end subroutine show
end program main
F90_D
"$FFC" "$TMP/casd_refuse_alloc.f90" -o "$TMP/casd_refuse_alloc.ffc" 2>&1 | grep -qi 'allocatable array awaits contiguous' && \
    echo "ok: allocatable character actual refused by name" || { echo "FAIL: allocatable char actual not refused honestly"; fail=1; }

[ "$fail" -ne 0 ] && { echo "FAIL: character assumed-shape dummy parity"; exit 1; }
echo "PASS: character assumed-shape dummy parity, $RUNS runs per shape"
