#!/usr/bin/env bash
# Oracle for general runtime character length expressions (task-3d/#348):
# character(len=len(x)+k) locals and function results, len(x)-k with
# truncation, negative-length clamping, and bare integer-variable lengths.
# All shapes byte-exact vs gfortran; procedures are called with multiple
# different actual lengths so no compile-time fold can satisfy the output.
set -u
FFC=${FFC:-$(cd "$(dirname "$0")/.." && pwd)/build/fo/bin/ffc}
RUNS=${RUNS:-20}
TMP=$(mktemp -d /var/tmp/ffc-charlen-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
fail=0

cat > "$TMP/result_len_offset.f90" <<'F90_A'
program main
    print '(a)', '['//greet('world')//']'
    print '(a)', '['//greet('hi')//']'
    print '(a)', '['//greet('fortran')//']'
contains
    function greet(who) result(s)
        character(len=*), intent(in) :: who
        character(len=len(who)+5) :: s
        s = 'hi '//who
    end function greet
end program main
F90_A

cat > "$TMP/local_len_offset.f90" <<'F90_B'
program main
    call t('world')
    call t('hey')
contains
    subroutine t(who)
        character(len=*), intent(in) :: who
        character(len=len(who)+5) :: s
        s = 'hi '//who
        print '(a)', '['//s//']'
    end subroutine t
end program main
F90_B

cat > "$TMP/local_len_minus.f90" <<'F90_C'
program main
    call t('world')
    call t('abcdefg')
contains
    subroutine t(who)
        character(len=*), intent(in) :: who
        character(len=len(who)-2) :: s
        s = 'abcdefgh'
        print '(a)', '['//s//']'
    end subroutine t
end program main
F90_C

cat > "$TMP/len_clamp_zero.f90" <<'F90_D'
program main
    call t('ab')
contains
    subroutine t(who)
        character(len=*), intent(in) :: who
        character(len=len(who)-10) :: s
        s = 'xyz'
        print '(a)', '['//s//']'
    end subroutine t
end program main
F90_D

cat > "$TMP/bare_int_len.f90" <<'F90_E'
program main
    integer :: n
    n = 6
    call t('world')
contains
    subroutine t(who)
        character(len=*), intent(in) :: who
        character(len=len(who)) :: s
        s = 'hi'
        print '(a)', '['//s//']'
    end subroutine t
end program main
F90_E

cat > "$TMP/external_char_prefix.f90" <<'F90_F'
program main
    implicit none
    interface
        character(1) function pick(i)
            integer, intent(in) :: i
        end function pick
        character(len=3) function abbr()
        end function abbr
    end interface
    print '(a)', '['//pick(1)//pick(-1)//']'
    print '(a)', '['//abbr()//']'
end program main

character(1) function pick(i)
    integer, intent(in) :: i
    if (i > 0) then
        pick = 'Y'
    else
        pick = 'N'
    end if
end function pick

character(len=3) function abbr()
    abbr = 'fortran'
end function abbr
F90_F

for name in result_len_offset local_len_offset local_len_minus len_clamp_zero bare_int_len external_char_prefix; do
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

[ "$fail" -ne 0 ] && { echo "FAIL: runtime character length parity"; exit 1; }
echo "PASS: runtime character length parity, $RUNS runs per shape"
