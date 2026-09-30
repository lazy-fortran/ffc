#!/usr/bin/env bash
# ELEMENTAL external reference parity (#1351).
#
# F2018 15.4.2.1: referencing an ELEMENTAL external subprogram requires an
# explicit interface. An EXTERNAL declaration is an implicit interface, so
# `real, external :: sq` called as sq(x) is invalid even with a scalar actual.
# ffc accepted it with no diagnostic while gfortran rejected it. The rule lives
# in fortfront (semantic_explicit_interface_checker.f90); it is observed here
# through ffc because ffc is what decides acceptance.
#
# PURE is a required POSITIVE, not an extra: gfortran does not demand an
# explicit interface for a pure external reference, so registering PURE beside
# ELEMENTAL turns this gate red. That is the guard's purpose.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFC="${FFC:-$ROOT/build/fo/bin/ffc}"
WORK="$(mktemp -d /tmp/ffc-elemental-iface.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail=0
cases=0
rc_g=0
rc_f=0

# fixture <tag>: write stdin into the case source, fail loudly if empty.
fixture() {
    local tag="$1" dir
    dir="$WORK/$tag"
    mkdir -p "$dir"
    cat > "$dir/$tag.f90"
    if [ ! -s "$dir/$tag.f90" ]; then
        echo "FAIL: fixture $tag.f90 is empty"
        fail=1
    fi
}

compare() {
    local tag="$1" dir="$WORK/$1"
    ( cd "$dir" && gfortran -std=f2018 -fsyntax-only "$tag.f90" ) >"$dir/g.out" 2>&1
    rc_g=$?
    ( cd "$dir" && timeout 60 "$FFC" "$tag.f90" -o "$dir/$tag.bin" ) >"$dir/f.out" 2>&1
    rc_f=$?
    cases=$((cases + 1))
}

# accept <tag> <desc>
accept() {
    compare "$1"
    if [ "$rc_g" -ne 0 ]; then
        echo "FAIL: gfortran rejects a case claimed valid: $2"
        grep -m2 -E "Error|error" "$WORK/$1/g.out"
        fail=1; return
    fi
    if [ "$rc_f" -ne 0 ]; then
        echo "FAIL: valid case rejected by ffc: $2"
        grep -m2 -E "error|Error" "$WORK/$1/f.out"
        fail=1; return
    fi
    echo "ok: $2"
}

# reject <tag> <desc> <needle>
reject() {
    compare "$1"
    if [ "$rc_g" -eq 0 ]; then
        echo "FAIL: gfortran accepts a case claimed invalid: $2"
        fail=1; return
    fi
    if [ "$rc_f" -eq 0 ]; then
        echo "FAIL: over-accept, ffc accepted: $2"
        fail=1; return
    fi
    if ! grep -qi "$3" "$WORK/$1/f.out"; then
        echo "FAIL: rejection did not name the reason ($3): $2"
        head -3 "$WORK/$1/f.out"
        fail=1; return
    fi
    echo "ok: $2"
}

# VALID: elemental reached through a module USE, applied to an array. This is
# how elemental procedures are meant to be called.
fixture elemental_via_use <<'ENDFIX'
module m_ok
    implicit none
contains
    elemental function sq(x) result(y)
        real, intent(in) :: x
        real :: y
        y = x * x
    end function sq
end module m_ok

program user
    use m_ok
    implicit none
    real :: a(3)
    a = [1.0, 2.0, 3.0]
    a = sq(a)
    print *, a
end program user
ENDFIX
accept elemental_via_use "ELEMENTAL via USE applied to an array"

# INVALID: the same elemental procedure reached through EXTERNAL, scalar actual.
fixture elemental_external <<'ENDFIX'
elemental function sq(x) result(y)
    real, intent(in) :: x
    real :: y
    y = x * x
end function sq

program user
    implicit none
    real, external :: sq
    real :: x
    x = 2.0
    print *, sq(x)
end program user
ENDFIX
reject elemental_external "ELEMENTAL called through EXTERNAL (scalar actual)" "elemental procedure"

# INVALID: elemental external with array actual - the #1351 shape.
fixture elemental_external_array <<'ENDFIX'
elemental function sq(x) result(y)
    real, intent(in) :: x
    real :: y
    y = x * x
end function sq

program user
    implicit none
    real, external :: sq
    real :: a(3)
    a = [1.0, 2.0, 3.0]
    a = sq(a)
    print *, a
end program user
ENDFIX
reject elemental_external_array "ELEMENTAL EXTERNAL with array actual" "elemental procedure"

# INVALID: elemental subroutine through EXTERNAL - the call-statement path,
# which is a subroutine_call_node and reaches the check by a different route.
fixture elemental_sub_external <<'ENDFIX'
elemental subroutine bump(x)
    integer, intent(inout) :: x
    x = x + 1
end subroutine bump

program user
    implicit none
    integer :: a
    a = 1
    call bump(a)
    print *, a
end program user
ENDFIX
reject elemental_sub_external "ELEMENTAL subroutine through EXTERNAL" "elemental procedure"

# VALID: PURE external through EXTERNAL. gfortran does not require an explicit
# interface here, so this pins the boundary of the fix.
fixture pure_external <<'ENDFIX'
pure function tri(a) result(y)
    integer, intent(in) :: a
    integer :: y
    y = a * 3
end function tri

program user
    implicit none
    integer, external :: tri
    print *, tri(5)
end program user
ENDFIX
accept pure_external "PURE external through EXTERNAL stays valid"

# VALID: ordinary non-elemental external through EXTERNAL.
fixture plain_external <<'ENDFIX'
function thrice(a) result(y)
    integer, intent(in) :: a
    integer :: y
    y = a * 3
end function thrice

program user
    implicit none
    integer, external :: thrice
    print *, thrice(5)
end program user
ENDFIX
accept plain_external "plain EXTERNAL reference stays valid"

# VALID: elemental reached through an explicit interface block.
fixture elemental_interface_block <<'ENDFIX'
program user
    implicit none
    interface
        elemental function sq(x) result(y)
            real, intent(in) :: x
            real :: y
        end function sq
    end interface
    real :: a(2)
    a = [1.0, 2.0]
    a = sq(a)
    print *, a
end program user

elemental function sq(x) result(y)
    real, intent(in) :: x
    real :: y
    y = x * x
end function sq
ENDFIX
accept elemental_interface_block "ELEMENTAL via explicit interface block"

# Corpus fixture that motivated the fix - required, not optional.
FIX="$ROOT/../fortfront/examples/f90/issue_1351_elemental_pure.f90"
if [ ! -f "$FIX" ]; then
    echo "FAIL: required corpus fixture missing: $FIX"
    fail=1
else
    cases=$((cases + 1))
    if ( cd "$WORK" && timeout 60 "$FFC" "$FIX" -o "$WORK/corpus.bin" ) >"$WORK/corpus.out" 2>&1; then
        echo "FAIL: corpus fixture issue_1351_elemental_pure.f90 still over-accepted"
        fail=1
    elif ! grep -qi "elemental procedure" "$WORK/corpus.out"; then
        echo "FAIL: corpus rejection did not name elemental"
        head -3 "$WORK/corpus.out"
        fail=1
    else
        echo "ok: issue_1351_elemental_pure.f90 rejected, naming elemental"
    fi
fi

if [ "$cases" -eq 0 ]; then
    echo "FAIL: no cases executed"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "elemental explicit-interface parity: PASS ($cases cases)"
else
    echo "elemental explicit-interface parity: FAILED ($cases cases)"
fi
exit "$fail"
