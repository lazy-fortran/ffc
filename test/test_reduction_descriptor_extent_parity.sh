#!/usr/bin/env bash
# Oracle for task-3e (#339 slice A): reductions and ubound over
# descriptor-bound assumed-shape dummies read extents from the caller's
# LIVE canonical descriptor (read_runtime_dim_extent), not from the
# compile-time metadata cache. Two-call VARYING-extent shapes decide:
# a compile-time fold cannot answer both calls. Byte-exact vs gfortran,
# RUNS>=20, md5-recorded binaries.
set -u
FFC=${FFC:-$(cd "$(dirname "$0")/.." && pwd)/build/fo/bin/ffc}
RUNS=${RUNS:-20}
TMP=$(mktemp -d /var/tmp/ffc-rre-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
fail=0

cat > "$TMP/rre_rank1.f90" <<'F90'
program p
    implicit none
    integer :: a(5), b(3)
    a = [1,2,3,4,5]
    b = [10,20,30]
    call check(a)
    call check(b)
contains
    subroutine check(x)
        integer :: x(:)
        print *, size(x), sum(x), ubound(x,1), count(x>2)
        print *, any(x>4), all(x>0), product(x)
    end subroutine check
end program p
F90

cat > "$TMP/rre_rank2.f90" <<'F90'
program p
    implicit none
    integer :: m(2,3), n(3,2)
    m = reshape([1,2,3,4,5,6], [2,3])
    n = reshape([7,8,9,10,11,12], [3,2])
    call check(m)
    call check(n)
contains
    subroutine check(x)
        integer :: x(:,:)
        print *, size(x,1), size(x,2), sum(x), ubound(x,2)
        print *, count(x>6), any(x>11), all(x>0)
    end subroutine check
end program p
F90

cat > "$TMP/rre_rank2c.f90" <<'F90'
program p
    implicit none
    integer :: m(2,3), n(4,2)
    m = reshape([1,2,3,4,5,6], [2,3])
    n = reshape([7,8,9,10,11,12,13,14], [4,2])
    call check(m)
    call check(n)
contains
    subroutine check(x)
        integer :: x(:,:)
        print *, sum(x), count(x > 6), any(x == 14), all(x /= 0)
    end subroutine check
end program p
F90

cat > "$TMP/rre_size.f90" <<'F90'
program p
    implicit none
    integer :: a(4), b(6)
    a = [1,2,3,4]
    b = [5,6,7,8,9,10]
    call check(a)
    call check(b)
contains
    subroutine check(x)
        integer :: x(:)
        print *, size(x), size(x,1), ubound(x,1)
        print *, maxval(x), minval(x), count(x > 4)
    end subroutine check
end program p
F90

for name in rre_rank1 rre_rank2 rre_rank2c rre_size; do
    gfortran -std=f2018 -o "$TMP/$name.gfortran" "$TMP/$name.f90" || { echo "FAIL: gfortran build $name"; fail=1; continue; }
    "$FFC" "$TMP/$name.f90" -o "$TMP/$name.ffc" || { echo "FAIL: ffc refused $name"; fail=1; continue; }
    want=$("$TMP/$name.gfortran" | tr '\n' '|')
    md5=$(md5sum "$TMP/$name.ffc" | awk '{print $1}')
    runs=0
    while [ "$runs" -lt "$RUNS" ]; do
        got=$("$TMP/$name.ffc" 2>&1 | tr '\n' '|')
        rc=$?
        if [ "$rc" -ne 0 ]; then echo "FAIL: $name crashed rc=$rc"; fail=1; break; fi
        if [ "$got" != "$want" ]; then
            echo "FAIL: $name run $((runs+1)) got [$got] want [$want]"; fail=1; break
        fi
        runs=$((runs+1))
    done
    [ "$runs" -eq "$RUNS" ] && echo "ok: $name matches gfortran on $runs/$runs runs"
    echo "    binary md5: $md5"
done

[ "$fail" -ne 0 ] && { echo "FAIL: descriptor-extent reduction parity"; exit 1; }
echo "PASS: descriptor-extent reduction parity, $RUNS runs per shape"
