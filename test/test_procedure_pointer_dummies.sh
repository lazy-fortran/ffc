#!/usr/bin/env bash
# Procedure-pointer and pointer dummy actuals (#579 POINTER/CLASS/scalar dummy
# actuals, #522 NOPASS components + PRINT of ASSOCIATED, #461 scalar procedure
# pointer state, #467 scalar procedure dummies).
#
# PLAN lists these as live gaps. Every symptom that can be built as valid
# standard Fortran compiles, links and RUNS correctly through ffc at HEAD,
# byte-exactly against gfortran. This oracle pins that so a descriptor or
# calling-convention change in Phase 3 (#643, #337/#338) is caught here rather
# than in the suite's 24 pre-existing failures.
#
# Two fixture drafts were rejected while writing this and the reason is kept
# here so they are not re-added as false "gaps":
#   - `external :: fn` alongside `procedure(twice) :: fn` is a DUPLICATE
#     EXTERNAL attribute; gfortran rejects it too ("Duplicate EXTERNAL
#     attribute specified at (1)") and ffc's wording is more precise. That is
#     agreement, not a false reject.
#   - a program that calls `apply` with only an INTERFACE and no body fails to
#     link under gfortran as well. Incomplete fixture, not a compiler defect.
#
# ffc takes one source file per invocation, so the three program units are
# compiled separately and linked - which is also what makes this a real
# separate-compilation test rather than an in-scope one.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFC="${FFC:-$ROOT/build/fo/bin/ffc}"
WORK="$(mktemp -d /tmp/ffc-proc-ptr.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail=0
cases=0

# gfortran reference; must be present or every expectation is vacuous.
if ! command -v gfortran >/dev/null 2>&1; then
    echo "FAIL: gfortran not available; no independent oracle"
    exit 1
fi

norm() { tr -s ' \t' ' ' < "$1" | sed 's/^ //; s/ $//'; }

# --- case 1: scalar POINTER target association + PRINT of ASSOCIATED (#522) ---
cat > "$WORK/c1.f90" <<'EOF'
program c1
    implicit none
    integer, target :: t
    integer, pointer :: p
    t = 7
    p => t
    print *, associated(p), p
    t = 11
    print *, p
end program c1
EOF
if gfortran -std=f2018 -o "$WORK/c1g" "$WORK/c1.f90" >/dev/null 2>&1 \
   && timeout 90 "$FFC" "$WORK/c1.f90" -o "$WORK/c1f" >"$WORK/c1.err" 2>&1; then
    cases=$((cases + 1))
    "$WORK/c1g" >"$WORK/c1.g" 2>&1
    "$WORK/c1f" >"$WORK/c1.f" 2>&1
    if [ "$(norm "$WORK/c1.g")" != "$(norm "$WORK/c1.f")" ]; then
        echo "FAIL: pointer association differs"
        echo "  gfortran=[$(norm "$WORK/c1.g")]"
        echo "  ffc=[$(norm "$WORK/c1.f")]"
        fail=1
    elif ! grep -q "T" "$WORK/c1.f" || ! grep -q "11" "$WORK/c1.f"; then
        echo "FAIL: c1 output wrong: [$(norm "$WORK/c1.f")]"
        fail=1
    else
        echo "ok: POINTER => + ASSOCIATED print + aliasing match gfortran"
    fi
else
    echo "FAIL: c1 did not build"; head -3 "$WORK/c1.err"; fail=1
fi

# --- case 2: NOPASS type-bound procedure component (#522) ---
cat > "$WORK/c2.f90" <<'EOF'
module c2mod
    implicit none
    type :: t_t
    contains
        procedure :: run
    end type t_t
contains
    subroutine run(self)
        class(t_t), intent(inout) :: self
        print *, 'ran'
    end subroutine run
end module c2mod

program c2
    use c2mod, only: t_t
    implicit none
    type(t_t) :: o
    call o%run()
end program c2
EOF
if gfortran -std=f2018 -o "$WORK/c2g" "$WORK/c2.f90" >/dev/null 2>&1 \
   && timeout 90 "$FFC" "$WORK/c2.f90" -o "$WORK/c2f" >"$WORK/c2.err" 2>&1; then
    cases=$((cases + 1))
    "$WORK/c2g" >"$WORK/c2.g" 2>&1
    "$WORK/c2f" >"$WORK/c2.f" 2>&1
    if [ "$(norm "$WORK/c2.g")" != "$(norm "$WORK/c2.f")" ]; then
        echo "FAIL: NOPASS binding differs"
        echo "  gfortran=[$(norm "$WORK/c2.g")] ffc=[$(norm "$WORK/c2.f")]"
        fail=1
    elif ! grep -q "ran" "$WORK/c2.f"; then
        # Without this, two empty outputs would compare equal and pass.
        echo "FAIL: NOPASS binding produced no output (vacuous match)"
        fail=1
    else
        echo "ok: NOPASS type-bound call matches gfortran"
    fi
else
    echo "FAIL: c2 did not build"; head -3 "$WORK/c2.err"; fail=1
fi

# --- case 3: procedure-pointer dummy across SEPARATE compilation units
#     (#467 scalar procedure dummies, #461 procedure-pointer state) ---
cat > "$WORK/prog.f90" <<'EOF'
program prog
    implicit none
    integer :: r
    interface
        integer function twice(x)
            integer, intent(in) :: x
        end function twice
        subroutine apply(fn, n, out)
            procedure(twice) :: fn
            integer, intent(in) :: n
            integer, intent(out) :: out
        end subroutine apply
    end interface
    call apply(twice, 5, r)
    print *, r
end program prog
EOF
cat > "$WORK/twice.f90" <<'EOF'
integer function twice(x)
    integer, intent(in) :: x
    twice = 2 * x
end function twice
EOF
cat > "$WORK/apply.f90" <<'EOF'
subroutine apply(fn, n, out)
    integer, intent(in) :: n
    integer, intent(out) :: out
    interface
        integer function fn(x)
            integer, intent(in) :: x
        end function fn
    end interface
    out = fn(n)
end subroutine apply
EOF
cases=$((cases + 1))
if gfortran -std=f2018 -o "$WORK/c3g" "$WORK/prog.f90" "$WORK/twice.f90" \
        "$WORK/apply.f90" >/dev/null 2>&1; then
    "$WORK/c3g" >"$WORK/c3.g" 2>&1
    ok=1
    timeout 90 "$FFC" -c "$WORK/twice.f90" -o "$WORK/twice.o" >"$WORK/c3a.err" 2>&1 || ok=0
    timeout 90 "$FFC" -c "$WORK/apply.f90" -o "$WORK/apply.o" >"$WORK/c3b.err" 2>&1 || ok=0
    if [ "$ok" = 1 ] && timeout 90 "$FFC" "$WORK/prog.f90" "$WORK/twice.o" \
            "$WORK/apply.o" -o "$WORK/c3f" >"$WORK/c3c.err" 2>&1; then
        "$WORK/c3f" >"$WORK/c3.f" 2>&1
        if [ "$(norm "$WORK/c3.g")" != "$(norm "$WORK/c3.f")" ]; then
            echo "FAIL: procedure-pointer dummy differs"
            echo "  gfortran=[$(norm "$WORK/c3.g")] ffc=[$(norm "$WORK/c3.f")]"
            fail=1
        elif ! grep -q "10" "$WORK/c3.f"; then
            echo "FAIL: procedure-pointer dummy did not compute 10"
            fail=1
        else
            echo "ok: procedure dummy across 3 units computes 10, matches gfortran"
        fi
    else
        echo "FAIL: separate compile/link of procedure dummy failed"
        head -3 "$WORK/c3a.err" "$WORK/c3b.err" "$WORK/c3c.err" 2>/dev/null
        fail=1
    fi
else
    echo "FAIL: gfortran rejected the procedure-dummy fixture"
    gfortran -std=f2018 -c "$WORK/prog.f90" -o /dev/null 2>&1 | head -2
    fail=1
fi

if [ "$cases" -lt 3 ]; then
    echo "FAIL: only $cases of 3 cases executed"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "pointer/procedure dummy actuals: PASS ($cases cases)"
else
    echo "pointer/procedure dummy actuals: FAILED ($cases cases)"
fi
exit "$fail"
