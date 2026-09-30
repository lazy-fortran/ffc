#!/usr/bin/env bash
# DATA zero-fill seen list must grow, not refuse.
#
# F2018 8.5.16 puts no limit on how many objects a DATA statement may cover.
# The lowering kept the symbol indices of distinct DATA arrays in a fixed
# integer :: seen(64) and, once it was full, refused the unit with
# "DATA zero-fill seen list overflow". That made a valid program fail to
# compile purely because of how many arrays it declared: 64 distinct arrays
# compiled, 65 did not, while gfortran accepted both.
#
# The seen list is now allocatable and doubles on demand. This test pins the
# boundary and, more importantly, checks the SEMANTICS past it: element 2 of
# every array is never DATA-initialised, so the zero-fill owed by 8.5.16 must
# still land, at array 65 and at array 130. Compile-acceptance alone would pass
# with the memset silently skipped.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFC="${FFC:-$ROOT/build/fo/bin/ffc}"
WORK="$(mktemp -d /tmp/ffc-data-seen-cap.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail=0
cases=0

# emit <file> <n> : n distinct integer arrays, DATA-initialise element 1,
# print element 1 and element 2 of every 17th array.
emit() {
    local fn="$1" n="$2" i
    {
        echo "program big"
        echo "    implicit none"
        for ((i = 1; i <= n; i++)); do echo "    integer :: a${i}(2)"; done
        for ((i = 1; i <= n; i++)); do echo "    data a${i}(1) /${i}/"; done
        for ((i = 1; i <= n; i += 17)); do
            echo "    print *, ${i}, a${i}(1), a${i}(2)"
        done
        echo "end program big"
    } > "$fn"
    if [ ! -s "$fn" ]; then
        echo "FAIL: fixture $fn not generated"
        fail=1
        return 1
    fi
    return 0
}

# parity <tag> <n> : both compilers compile, run, and agree byte-exactly.
parity() {
    local tag="$1" n="$2" src gf ff outrun
    src="$WORK/$tag.f90"
    emit "$src" "$n" || return
    cases=$((cases + 1))
    if ! gfortran -std=f2018 -o "$WORK/g_$tag" "$src" >"$WORK/g_$tag.build" 2>&1; then
        echo "FAIL: gfortran rejects a program it must accept ($tag, $n arrays)"
        head -3 "$WORK/g_$tag.build"
        fail=1; return
    fi
    if ! timeout 120 "$FFC" "$src" -o "$WORK/f_$tag" >"$WORK/f_$tag.build" 2>&1; then
        echo "FAIL: ffc rejects valid code ($tag, $n distinct DATA arrays)"
        grep -m2 -E "error|Error" "$WORK/f_$tag.build"
        fail=1; return
    fi
    "$WORK/g_$tag" > "$WORK/g_$tag.out" 2>&1
    "$WORK/f_$tag" > "$WORK/f_$tag.out" 2>&1
    if ! diff -q "$WORK/g_$tag.out" "$WORK/f_$tag.out" >/dev/null; then
        echo "FAIL: output differs ($tag, $n arrays)"
        diff "$WORK/g_$tag.out" "$WORK/f_$tag.out" | head -6
        fail=1; return
    fi
    # The zero-fill owed by 8.5.16 must actually be present in the output:
    # every printed line is <index> <element 1 from DATA> <element 2, which
    # must be 0>, so the last field of at least one line has to be 0.
    if ! awk 'NF>=3 && $NF==0 {found=1} END{exit found?0:1}' "$WORK/f_$tag.out"; then
        echo "FAIL: no zero-filled second element observed ($tag, $n arrays)"
        head -2 "$WORK/f_$tag.out"
        fail=1; return
    fi
    outrun=$(grep -c '' "$WORK/f_$tag.out")
    echo "ok: $tag ($n distinct DATA arrays) matches gfortran, zero-fill present"
}

# Old boundary must stay green; this is the last size the fixed array handled.
parity at_cap_64 64
# One past the old cap - the exact false reject this fixes.
parity past_cap_65 65
# Well past it, across two doublings of the seen list.
parity far_past_130 130
parity far_past_300 300

# Genuine negative: a named constant may not appear in DATA. The cap must not
# have been "fixed" by relaxing real rules, and the diagnostic must still name
# the offence and carry a position rather than landing at 1:1.
cat > "$WORK/neg_param.f90" <<'EOF'
program neg
    implicit none
    integer, parameter :: k = 7
    integer :: v(2)
    data k /9/
    print *, k, v(1)
end program neg
EOF
cases=$((cases + 1))
if gfortran -std=f2018 -fsyntax-only "$WORK/neg_param.f90" >/dev/null 2>&1; then
    echo "FAIL: gfortran accepts a program this test claims invalid"
    fail=1
elif timeout 60 "$FFC" "$WORK/neg_param.f90" -o "$WORK/neg.bin" >"$WORK/neg.out" 2>&1; then
    echo "FAIL: named constant in DATA was accepted"
    fail=1
elif ! grep -qi "named constant" "$WORK/neg.out"; then
    echo "FAIL: DATA rejection did not name the named constant"
    head -3 "$WORK/neg.out"
    fail=1
else
    echo "ok: named constant in DATA still rejected, naming the offence"
fi

if [ "$cases" -eq 0 ]; then
    echo "FAIL: no cases executed"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo "DATA seen-list growth: PASS ($cases cases)"
else
    echo "DATA seen-list growth: FAILED ($cases cases)"
fi
exit "$fail"
