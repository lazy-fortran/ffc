#!/usr/bin/env bash
# LOGICAL array elements and sections must match gfortran byte-exactly, on
# EVERY run (#337 arraydesc family).
#
# Why this file exists: the defect it pins had no test at all, which is why it
# shipped. A LOGICAL array written from a comparison inside a loop stored ONE
# byte into a FOUR-byte slot while the matching load read FOUR bytes, so the
# upper three bytes of every slot were uninitialised stack memory. A .false.
# element therefore came back as whatever garbage sat there, was nonzero, and
# printed .true. - nondeterministically. The same binary printed `F T T T T T`
# on most runs and the correct `F T F T F T` on a few, so a test that ran the
# program once had a decent chance of passing while the compiler was broken.
#
# Two independent halves, both found by experiment and both pinned here:
#   1. store/load WIDTH. liric's LR_OP_STORE took its width from the VALUE
#      operand (`size_t store_sz = lr_type_size(ops[0].type)`), and ffc's
#      emit_store_i32 delegated to emit_store_typed, which sets
#      inst%typ = c_null_ptr and takes no type argument at all. The compiler
#      documents at session_program_lowering_array_elements.inc that "a logical
#      element occupies an i32 slot (0/1)"; the store did not honour that.
#   2. PRINT format. emit_array_section_print_items dispatched F32/F64/else ->
#      i32 with NO logical case, so `print *, l(1:6:1)` emitted `0 1 0 1`
#      while `print *, l` and `print *, l(2)` printed `F T F T` correctly.
#
# The repetition is the point: RUNS>=20 per shape, one md5-recorded binary, so
# an uninitialised-memory regression cannot slip through on a lucky run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFC="${FFC:-$ROOT/build/fo/bin/ffc}"
RUNS="${RUNS:-20}"
WORK="$(mktemp -d /tmp/ffc-logical-parity.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail=0

# Every shape below is valid F2018 and compiles under gfortran; each is written
# to a fixture and guarded, because an empty fixture compiles and exits 0 and
# would report a vacuous pass.
check_shape() {
    local tag="$1"
    cat > "$WORK/$tag.f90"
    if [ ! -s "$WORK/$tag.f90" ]; then
        echo "FAIL: fixture $tag.f90 is empty"
        fail=1
        return
    fi
    if ! gfortran -std=f2018 -o "$WORK/$tag.g" "$WORK/$tag.f90" \
            > "$WORK/$tag.gf.log" 2>&1; then
        echo "FAIL: gfortran refused $tag.f90 (fixture is not valid F2018)"
        sed -n '1,4p' "$WORK/$tag.gf.log"
        fail=1
        return
    fi
    local expected
    expected="$("$WORK/$tag.g" | tr -d '[:space:]')"
    if [ -z "$expected" ]; then
        echo "FAIL: gfortran produced no output for $tag.f90"
        fail=1
        return
    fi
    if ! timeout 120 "$FFC" "$WORK/$tag.f90" -o "$WORK/$tag.f" \
            > "$WORK/$tag.f.log" 2>&1; then
        echo "FAIL: ffc refused $tag.f90"
        grep -oE 'error: .*' "$WORK/$tag.f.log" | head -2
        fail=1
        return
    fi
    local i actual bad=0
    for i in $(seq 1 "$RUNS"); do
        actual="$("$WORK/$tag.f" | tr -d '[:space:]')"
        [ "$actual" = "$expected" ] || bad=$((bad + 1))
    done
    if [ "$bad" -ne 0 ]; then
        echo "FAIL: $tag differed on $bad of $RUNS runs"
        echo "      expected: [$expected]"
        echo "      got:      [$actual]"
        fail=1
    else
        echo "ok: $tag matches gfortran on $RUNS/$RUNS runs [$expected]"
    fi
}

# The original repro: write from a comparison in one loop, read in the next.
# This is the shape that flipped .false. to .true. on most runs.
check_shape element_written_from_comparison <<'EOF'
program p
    implicit none
    logical :: l(6)
    integer :: i
    do i = 1, 6
        l(i) = (mod(i, 2) == 0)
    end do
    do i = 1, 6
        print *, l(i)
    end do
end program p
EOF

# Two arrays, different comparisons, same loop body. Added because it is the
# probe that localised the defect: only the FIRST store in the body was flaky,
# while the second stayed correct, which ruled out a shared temporary.
check_shape two_arrays_same_loop <<'EOF'
program p
    implicit none
    logical :: a(4), b(4)
    integer :: i
    do i = 1, 4
        a(i) = (i > 1)
        b(i) = (i > 3)
    end do
    print *, a(1), a(2), a(3), a(4)
    print *, b(1), b(2), b(3), b(4)
end program p
EOF

# All four kinds: the defect was kind-INdependent, so all four are pinned.
for k in 1 2 4 8; do
    check_shape "kind_$k" <<EOF
program p
    implicit none
    logical($k) :: l(4)
    integer :: i
    do i = 1, 4
        l(i) = (i > 2)
    end do
    print *, l(1), l(2), l(3), l(4)
end program p
EOF
done

# Wholes, sections and scalars: the whole-array and scalar forms were always
# right and the section form was always wrong, so all three are pinned here to
# keep the fix from drifting back to the half that used to work.
check_shape whole_scalar_and_sections <<'EOF'
program p
    implicit none
    logical :: l(6)
    integer :: i
    do i = 1, 6
        l(i) = (mod(i, 2) == 0)
    end do
    print *, l(2)
    print *, l
    print *, l(1:6:1)
    print *, l(2:6:2)
end program p
EOF

# A constant .false. store was correct even while the comparison store was not;
# kept as a control so a future width change cannot break it unnoticed.
check_shape constant_false_control <<'EOF'
program p
    implicit none
    logical :: l(4)
    integer :: i
    l = .true.
    do i = 1, 4
        l(i) = .false.
    end do
    print *, l(1), l(2), l(3), l(4)
end program p
EOF

# Integers share the i32 store that was changed, so they are pinned too: the
# fix must not have traded the working path for the broken one.
check_shape integer_store_unchanged <<'EOF'
program p
    implicit none
    integer :: n(6)
    integer :: i
    do i = 1, 6
        n(i) = mod(i, 2)
    end do
    do i = 1, 6
        print *, n(i)
    end do
end program p
EOF

if [ "$fail" -ne 0 ]; then
    echo "FAIL: logical array parity"
    exit 1
fi
echo "PASS: logical array parity, $RUNS runs per shape"
