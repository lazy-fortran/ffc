#!/usr/bin/env bash
# ffc#753 oracle: a SCALAR-result function called inside PRINT must be one
# correct call, not an elementwise expansion over the actual's shape.
# Before the fix, whole-array-print classification claimed any call whose
# ACTUAL was array-shaped, re-called the function once per element with
# scratch arguments, and printed phantom values (or segfaulted when the
# scratch was dereferenced as a descriptor). Every shape here must match
# gfortran byte-exact, and the side-effect shape must evaluate exactly once.
set -u
FFC=${FFC:-$(cd "$(dirname "$0")/.." && pwd)/build/fo/bin/ffc}
RUNS=${RUNS:-20}
TMP=$(mktemp -d /var/tmp/ffc-print-call-XXXXXX)
trap 'rm -rf "$TMP"' EXIT
fail=0

check_shape() {
    local name=$1; shift
    cat > "$TMP/$name.f90"
    gfortran -std=f2018 -o "$TMP/$name.gfortran" "$TMP/$name.f90" || {
        echo "FAIL: gfortran could not build $name"; fail=1; return; }
    "$FFC" "$TMP/$name.f90" -o "$TMP/$name.ffc" || {
        echo "FAIL: ffc refused $name"; fail=1; return; }
    local got_want md5runs=0
    want=$("$TMP/$name.gfortran")
    md5=$(md5sum "$TMP/$name.ffc" | awk '{print $1}')
    local runs=0
    while [ "$runs" -lt "$RUNS" ]; do
        got=$("$TMP/$name.ffc" 2>&1) || { echo "FAIL: $name crashed"; fail=1; return; }
        [ "$got" = "$want" ] || {
            echo "FAIL: $name run $((runs+1)) got [$got] want [$want]"
            echo "      binary md5: $md5"; fail=1; return; }
        runs=$((runs+1))
    done
    echo "ok: $name matches gfortran on $runs/$runs runs [$(echo "$want" | tr '\n' '|')]"
    echo "    binary md5: $md5"
}

check_shape module_print_assumed_scalar <<'EOF'
module sm
contains
function f(a) result(s)
integer, intent(in) :: a(:)
integer :: s
s = a(1) + a(2)
end function
end module
program p
use sm
implicit none
integer :: x(2)
x(1) = 10; x(2) = 20
print *, f(x)
end program
EOF

check_shape contained_print_assumed_scalar <<'EOF'
program p
implicit none
integer :: x(4)
x(1) = 10; x(2) = 20; x(3) = 30; x(4) = 40
print *, ssum(x)
contains
function ssum(a) result(s)
integer, intent(in) :: a(:)
integer :: s
s = a(1) + a(2)
end function
end program
EOF

check_shape print_single_evaluation_side_effect <<'EOF'
program p
implicit none
integer :: x(2), n
x(1) = 10; x(2) = 20; n = 0
print *, f(x)
print *, n
contains
function f(a) result(s)
integer :: a(:), s
s = a(1) + a(2)
n = n + 1
end function
end program
EOF

check_shape print_multi_entity_decl_scalar_result <<'EOF'
program p
implicit none
integer :: x(2)
x = 3
print *, g(x)
contains
function g(a) result(res)
integer :: a(2), res
res = 7
end function
end program
EOF

check_shape print_explicit_shape_trailing_scalar <<'EOF'
program p
implicit none
integer :: x(2)
x(1) = 3; x(2) = 4
print *, g(x)
contains
function g(a) result(s)
integer :: a(2), s
s = a(1) + a(2)
end function
end program
EOF

check_shape module_print_mixed_items <<'EOF'
module sm2
contains
function total(a) result(s)
integer, intent(in) :: a(:)
integer :: s
integer :: i
s = 0
do i = 1, size(a)
    s = s + a(i)
end do
end function
end module
program p
use sm2
implicit none
integer :: x(3), y(2)
x = (/1, 2, 3/)
y = (/10, 20/)
print *, total(x), total(y), total(x) + 1
end program
EOF

if [ "$fail" -ne 0 ]; then
    echo "FAIL: print scalar-result call parity"
    exit 1
fi
echo "PASS: print scalar-result call parity, $RUNS runs per shape"
