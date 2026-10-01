#!/usr/bin/env bash
# Oracle for assumed-rank SELECT RANK dynamic dispatch (task-3c/#337):
# one procedure called with different-rank actuals must select the matching
# arm per call, byte-exact vs gfortran. Empty SELECT RANK is a no-op
# (F2018 11.1.2.6); RANK (*) and rank>4 stay refused (visible diagnostics).
set -u
FFC=${FFC:-$(cd "$(dirname "$0")/.." && pwd)/build/fo/bin/ffc}
RUNS=${RUNS:-20}
TMP=$(mktemp -d /var/tmp/ffc-selectrank-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
fail=0

cat > "$TMP/multi_arm_dispatch.f90" <<'F90_MULTI'
program main
    implicit none
    real :: b(2), c(2,2), d(2,2,2)
    b = 1.0; c = 2.0; d = 3.0
    call work(b)
    call work(c)
    call work(d)
    call work(c)
contains
    subroutine work(x)
        real :: x(..)
        select rank (x)
        rank (1)
            print *, 11
        rank (2)
            print *, 22
        rank (3)
            print *, 33
        rank default
            print *, -1
        end select
    end subroutine work
end program main
F90_MULTI

cat > "$TMP/empty_select_rank_noop.f90" <<'F90_EMPTY'
program main
    implicit none
    real :: a(2,2,2)
    call work(a)
    print *, 7
contains
    subroutine work(x)
        real :: x(..)
        select rank (x)
        end select
    end subroutine work
end program main
F90_EMPTY

cat > "$TMP/default_only_dispatch.f90" <<'F90_DEFAULT'
program main
    implicit none
    real :: a(2,2,2,2)
    call work(a)
contains
    subroutine work(x)
        real :: x(..)
        select rank (x)
        rank default
            print *, -1
        end select
    end subroutine work
end program main
F90_DEFAULT

for name in multi_arm_dispatch empty_select_rank_noop default_only_dispatch; do
    gfortran -std=f2018 -o "$TMP/$name.gfortran" "$TMP/$name.f90" || { echo "FAIL: gfortran build $name"; fail=1; continue; }
    "$FFC" "$TMP/$name.f90" -o "$TMP/$name.ffc" || { echo "FAIL: ffc refused $name"; fail=1; continue; }
    want=$("$TMP/$name.gfortran")
    md5=$(md5sum "$TMP/$name.ffc" | awk '{print $1}')
    runs=0
    while [ "$runs" -lt "$RUNS" ]; do
        got=$("$TMP/$name.ffc" 2>&1) || { echo "FAIL: $name crashed"; fail=1; break; }
        if [ "$got" != "$want" ]; then
            echo "FAIL: $name run $((runs+1)) got [$got] want [$want]"
            echo "      binary md5: $md5"; fail=1; break
        fi
        runs=$((runs+1))
    done
    [ "$runs" -eq "$RUNS" ] && echo "ok: $name matches gfortran on $runs/$runs runs [$(echo "$want" | tr '\n' '|')]"
    echo "    binary md5: $md5"
done

# refusals stay visible: rank (*) and rank (5)
printf 'program main\n  real :: a(2,2)\n  call work(a)\ncontains\n  subroutine work(x)\n    real :: x(..)\n    select rank (x)\n    rank (*)\n      print *, 1\n    end select\n  end subroutine work\nend program main\n' > "$TMP/refuse_star.f90"
printf 'program main\n  real :: a(2,2,2,2,2)\n  call work(a)\ncontains\n  subroutine work(x)\n    real :: x(..)\n    select rank (x)\n    rank (5)\n      print *, 55\n    rank default\n      print *, -1\n    end select\n  end subroutine work\nend program main\n' > "$TMP/refuse_rank5.f90"
for spec in star rank5; do
    if "$FFC" "$TMP/refuse_$spec.f90" -o "$TMP/refuse_$spec.exe" 2>"$TMP/refuse_$spec.err"; then
        echo "FAIL: $spec accepted but must stay refused"; fail=1
    else
        grep -qi "refused\|are supported" "$TMP/refuse_$spec.err" || { echo "FAIL: $spec diagnostic missing refusal text"; fail=1; }
    fi
done

[ "$fail" -ne 0 ] && { echo "FAIL: select rank dispatch parity"; exit 1; }
echo "PASS: select rank dispatch parity, $RUNS runs per shape"
