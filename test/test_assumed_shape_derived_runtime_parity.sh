#!/usr/bin/env bash
# Assumed-shape DERIVED dummies read their extent from the caller's array
# descriptor, so a runtime-sized actual (allocatable, pointer, or a differently
# sized second call) must match gfortran byte-exactly on EVERY run (#337).
#
# Why this file exists: the guard in derived_assumed_shape_extents required the
# assumed-shape derived dummy's extent to fold to a compile-time constant from a
# whole-array actual, and rejected every other shape with
#   "assumed-shape derived dummy extent must come from a whole-array actual of
#    compile-time size". But the dummy was ALREADY bound through the caller's
# descriptor (#334): bind_assumed_shape_descriptor_params had loaded the base
# address and each runtime extent from it. For a rank-1 derived dummy the element
# offset is (sub-lower)*slot_width off the descriptor base, so no compile-time
# extent is needed and the guard was an over-rejection of valid F2018.
#
# The decisive shape is `two_calls_different_sizes`: the SAME callee called twice
# with actuals of extent 3 then 2. A compile-time specialization cannot answer
# both; the descriptor must drive the shape and each call must see its OWN
# extent. If the dummy had kept a folded stride, the second call would read past
# or miss elements and the sum would be wrong - nondeterministically, since it
# reads live descriptor memory. RUNS>=20 pins it against a lucky run.
#
# Rank>=2 monomorphic dummies now linearise with descriptor extents. The
# polymorphic class(t) dummy and the strided-section actuals keep their own
# contracts (#422) and are NOT in this file.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFC="${FFC:-$ROOT/build/fo/bin/ffc}"
RUNS="${RUNS:-20}"
if ! [ "$RUNS" -eq "$RUNS" ] 2>/dev/null || [ "$RUNS" -lt 20 ]; then
    echo "note: RUNS=$RUNS is below the parity floor; using 20"
    RUNS=20
fi
WORK="$(mktemp -d /tmp/ffc-asdr-runtime.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail=0

check_shape() {
    local tag="$1"
    cat > "$WORK/$tag.f90"
    if [ ! -s "$WORK/$tag.f90" ]; then
        echo "FAIL: fixture $tag.f90 is empty"; fail=1; return
    fi
    if ! gfortran -std=f2018 -o "$WORK/$tag.g" "$WORK/$tag.f90" \
            > "$WORK/$tag.gf.log" 2>&1; then
        echo "FAIL: gfortran refused $tag.f90 (not valid F2018)"
        sed -n '1,4p' "$WORK/$tag.gf.log"; fail=1; return
    fi
    local expected gfort_status
    expected="$("$WORK/$tag.g" | tr -d '[:space:]')"
    gfort_status=$?
    if [ "$gfort_status" -ne 0 ]; then
        echo "FAIL: gfortran binary for $tag exited $gfort_status"; fail=1; return
    fi
    if [ -z "$expected" ]; then
        echo "FAIL: gfortran produced no output for $tag.f90"; fail=1; return
    fi
    if ! timeout 120 "$FFC" "$WORK/$tag.f90" -o "$WORK/$tag.f" \
            > "$WORK/$tag.f.log" 2>&1; then
        echo "FAIL: ffc refused $tag.f90"
        grep -oE 'error: .*' "$WORK/$tag.f.log" | head -2
        fail=1; return
    fi
    local binary_md5
    binary_md5="$(md5sum "$WORK/$tag.f" | cut -d' ' -f1)"
    local i actual bad=0 crashed=0 run_status
    for i in $(seq 1 "$RUNS"); do
        actual="$("$WORK/$tag.f" 2>"$WORK/$tag.run.$i.err" | tr -d '[:space:]')"
        run_status=$?
        if [ "$run_status" -ne 0 ]; then crashed=$((crashed+1)); continue; fi
        [ "$actual" = "$expected" ] || bad=$((bad+1))
    done
    if [ "$crashed" -ne 0 ]; then
        echo "FAIL: $tag binary exited nonzero on $crashed of $RUNS runs"
        sed -n '1,2p' "$WORK/$tag.run.1.err" 2>/dev/null
        echo "      binary md5: $binary_md5"; fail=1
    fi
    if [ "$bad" -ne 0 ]; then
        echo "FAIL: $tag differed on $bad of $RUNS runs"
        echo "      expected: [$expected]"; echo "      got:      [$actual]"
        echo "      binary md5: $binary_md5"; fail=1
    elif [ "$crashed" -eq 0 ]; then
        echo "ok: $tag matches gfortran on $RUNS/$RUNS runs [$expected]"
        echo "    binary md5: $binary_md5"
    fi
}

# The decisive case: one callee, two actuals of extent 3 then 2. The descriptor
# must supply each call's own extent. A folded compile-time stride answers only
# one of them.
check_shape two_calls_different_sizes <<'EOF'
module m
    type :: pt
        integer :: x
    end type pt
contains
    subroutine total(a)
        type(pt), intent(in) :: a(:)
        integer :: i, s
        s = 0
        do i = 1, size(a)
            s = s + a(i)%x
        end do
        print *, s
    end subroutine total
end module m
program p
    use m
    implicit none
    type(pt), allocatable :: g(:)
    type(pt) :: h(2)
    allocate(g(3))
    g(1)%x = 1; g(2)%x = 2; g(3)%x = 3
    h(1)%x = 10; h(2)%x = 20
    call total(g)
    call total(h)
end program p
EOF

# Whole-array allocatable actual (runtime extent), the direct repro.
check_shape allocatable_whole_actual <<'EOF'
module m
    type :: pt
        integer :: x
    end type pt
contains
    subroutine total(a)
        type(pt), intent(in) :: a(:)
        integer :: i, s
        s = 0
        do i = 1, size(a)
            s = s + a(i)%x
        end do
        print *, s
    end subroutine total
end module m
program p
    use m
    implicit none
    type(pt), allocatable :: g(:)
    integer :: k
    allocate(g(5))
    do k = 1, 5
        g(k)%x = k * 10
    end do
    call total(g)
end program p
EOF

# INTENT(INOUT) component write through a runtime-extent descriptor actual.
check_shape allocatable_inout_component_write <<'EOF'
module m
    type :: pt
        integer :: x
    end type pt
contains
    subroutine bump(a, by)
        type(pt), intent(inout) :: a(:)
        integer, intent(in) :: by
        integer :: i
        do i = 1, size(a)
            a(i)%x = a(i)%x + by
        end do
    end subroutine bump
end module m
program p
    use m
    implicit none
    type(pt), allocatable :: g(:)
    integer :: k
    allocate(g(4))
    do k = 1, 4
        g(k)%x = k
    end do
    call bump(g, 100)
    do k = 1, 4
        print *, g(k)%x
    end do
end program p
EOF

# Fixed-shape whole-array actual control: this worked before the relaxation
# and must keep working, so the fix cannot have traded it away.
check_shape fixed_shape_control <<'EOF'
module m
    type :: pt
        integer :: x
    end type pt
contains
    subroutine total(a)
        type(pt), intent(in) :: a(:)
        integer :: i, s
        s = 0
        do i = 1, size(a)
            s = s + a(i)%x
        end do
        print *, s
    end subroutine total
end module m
program p
    use m
    implicit none
    type(pt) :: h(3)
    h(1)%x = 1; h(2)%x = 2; h(3)%x = 3
    call total(h)
end program p
EOF

# Rank>=2 assumed-shape derived dummies are covered: rank-2 columns scale
# the running stride by the leading extent READ FROM THE DESCRIPTOR at run
# time, so allocatable and fixed actuals of different shapes each work.
check_shape rank2_allocatable_actual <<'EOF'
module m
    type :: pt
        integer :: x
    end type pt
contains
    subroutine totals(a)
        type(pt), intent(in) :: a(:,:)
        integer :: i, j, s
        s = 0
        do j = 1, size(a, 2)
            do i = 1, size(a, 1)
                s = s + a(i,j)%x
            end do
        end do
        print *, s, size(a,1), size(a,2)
    end subroutine totals
end module m
program p
    use m
    implicit none
    integer :: i, j
    type(pt), allocatable :: g(:,:)
    type(pt) :: h(2,2)
    allocate(g(2,3))
    do j = 1, 3
        do i = 1, 2
            g(i,j)%x = i + 10*j
        end do
    end do
    call totals(g)
    h(1,1)%x = 1; h(2,1)%x = 2; h(1,2)%x = 3; h(2,2)%x = 4
    call totals(h)
end program p
EOF

# Rank-2 INTENT(INOUT) element write through a runtime leading extent: the
# write must land on exactly the addressed element of the allocatable actual.
check_shape rank2_inout_element_write <<'EOF'
module m
    type :: pt
        integer :: x
    end type pt
contains
    subroutine touch(a, i, j)
        type(pt), intent(inout) :: a(:,:)
        integer, intent(in) :: i, j
        a(i,j)%x = i*10 + j
    end subroutine touch
end module m
program p
    use m
    implicit none
    integer :: n
    type(pt), allocatable :: g(:,:)
    n = 2
    allocate(g(n,3))
    call touch(g, 2, 3)
    print *, g(2,3)%x, g(1,1)%x
end program p
EOF

# Nested component write through a rank-2 assumed-shape derived dummy:
# items(i,j)%payload%value must land on the addressed element, with the
# leading extent read from the descriptor (allocatable actual) and from a
# compile-time shape (fixed actual).
check_shape rank2_nested_component_write <<'EOF'
module m
    type :: payload_t
        integer :: value
    end type payload_t
    type :: item_t
        type(payload_t) :: payload
    end type item_t
contains
    subroutine ex(items, i, j)
        type(item_t), intent(inout) :: items(:,:)
        items(i,j)%payload%value = i*10 + j
    end subroutine ex
end module m
program p
    use m
    implicit none
    type(item_t), allocatable :: g(:,:)
    type(item_t) :: h(2,2)
    allocate(g(2,3))
    call ex(g, 2, 3)
    call ex(h, 1, 2)
    print *, g(2,3)%payload%value
    h(1,1)%payload%value = -1
    print *, h(1,2)%payload%value, h(1,1)%payload%value
end program p
EOF

if [ "$fail" -ne 0 ]; then
    echo "FAIL: assumed-shape derived runtime-extent parity"; exit 1
fi
echo "PASS: assumed-shape derived runtime-extent parity, $RUNS runs per shape"
