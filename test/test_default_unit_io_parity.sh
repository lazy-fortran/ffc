#!/usr/bin/env bash
# Default-file (fort.<N>) I/O parity with gfortran.
#
# A numeric unit that was never OPENed gets the default file fort.<N>, and the
# first statement that touches the unit decides the fate of that file. ffc got
# both halves wrong, in the loudest possible way:
#
#   READ(unit=10,fmt='(a)') into fort.10 holding "world" printed s=[] because
#   the runtime opened fort.10 "w+", which truncated it. The program's data was
#   destroyed by the act of reading it, and the run exited 0.
#
#   A READ that reaches end of file with none of END=, ERR= or IOSTAT= is an
#   error condition: gfortran prints "Fortran runtime error: End of file" and
#   exits 2. ffc fell through, printed the untouched variable, and exited 0 --
#   a wrong answer that looked like a right one.
#
# WRITE must keep truncating, which is what gfortran does when a WRITE makes
# the connection, so the mode follows the intent of the first statement rather
# than being a fixed choice.
#
# Each case runs in its own scratch directory so both compilers see byte-ident
# ical starting file state, and exit status is compared, not just stdout.
#
# Exit: 0 parity holds, 1 parity broken, 2 environment problem.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRATCH_ROOT=${TMPDIR:-/var/tmp/ert}
mkdir -p "$SCRATCH_ROOT"
WORK=$(mktemp -d "$SCRATCH_ROOT/ffc-default-unit-io.XXXXXX")
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

FFC=${FFC_DEFAULT_UNIT_IO_FFC:-$ROOT/build/fo/bin/ffc}
[ -x "$FFC" ] || { echo "default-unit-io: ffc not built at $FFC" >&2; exit 2; }
command -v gfortran >/dev/null || { echo "default-unit-io: gfortran required" >&2; exit 2; }

fail=0

# run_case <name> <source-file> <setup-script-or-"">
# Builds the same source with both compilers, gives each an identical scratch
# dir prepared by setup, runs both, and compares stdout, stderr presence and
# exit status.
run_case() {
    local name=$1 src=$2 setup=$3
    local gd=$WORK/$name.g fd=$WORK/$name.f
    local gdir=$WORK/$name.gdir fdir=$WORK/$name.fdir
    local gs=$WORK/$name.g.out fs=$WORK/$name.f.out
    local ge=$WORK/$name.g.err fe=$WORK/$name.f.err
    local rcg rcf

    mkdir -p "$gdir" "$fdir"
    if ! gfortran -o "$gd" "$src" >"$WORK/$name.gbuild" 2>&1; then
        echo "SKIP: $name - gfortran cannot build it (no reference)"
        return 0
    fi
    if ! "$FFC" "$src" -o "$fd" >"$WORK/$name.fbuild" 2>&1; then
        echo "FAIL: $name - ffc build failed while gfortran succeeded"
        tail -5 "$WORK/$name.fbuild"
        fail=1
        return 0
    fi

    if [ -n "$setup" ]; then
        ( cd "$gdir" && eval "$setup" )
        ( cd "$fdir" && eval "$setup" )
    fi

    ( cd "$gdir" && timeout 20 "$gd" ) >"$gs" 2>"$ge"
    rcg=$?
    ( cd "$fdir" && timeout 20 "$fd" ) >"$fs" 2>"$fe"
    rcf=$?

    # stdout is the program's answer and is compared byte for byte.
    if [ "$rcg" -ne "$rcf" ]; then
        echo "FAIL: $name exit status differs (gfortran=$rcg ffc=$rcf)"
        fail=1
    fi
    if ! diff -u "$gs" "$fs" >"$WORK/$name.diff"; then
        echo "FAIL: $name stdout differs"
        head -10 "$WORK/$name.diff"
        fail=1
    fi

    # stderr is normalised before comparison. gfortran prefixes its I/O errors
    # with an "At line N of file <path>" line and then dumps a backtrace whose
    # frames are load addresses, so those bytes differ between runs and between
    # frontends by construction and can never be a byte-exact oracle. What must
    # agree is the substantive diagnostic, so the noise is filtered and the
    # remaining lines are compared exactly.
    grep -vE '^At line |^Error termination|^#[0-9]|^[[:space:]]*$' "$ge" \
        >"$WORK/$name.ge.norm" || true
    grep -vE '^At line |^Error termination|^#[0-9]|^[[:space:]]*$' "$fe" \
        >"$WORK/$name.fe.norm" || true
    if ! diff -u "$WORK/$name.ge.norm" "$WORK/$name.fe.norm" \
            >"$WORK/$name.errdiff"; then
        echo "FAIL: $name diagnostic differs"
        head -8 "$WORK/$name.errdiff"
        fail=1
    fi

    # The file left behind is part of the observable behaviour: a truncated
    # default file and a preserved one are different results. Only files that
    # both sides leave are content-compared; a program that creates a default
    # file the other does not is reported, because default-file creation on
    # error termination is not yet pinned either way.
    local f base
    for f in "$gdir"/*; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        if [ -e "$fdir/$base" ]; then
            if ! diff -q "$f" "$fdir/$base" >/dev/null; then
                echo "FAIL: $name left different content in $base"
                echo "  gfortran: [$(head -c 40 "$f" | tr '\n' '|')]"
                echo "  ffc:      [$(head -c 40 "$fdir/$base" | tr '\n' '|')]"
                fail=1
            fi
        else
            echo "NOTE: gfortran left $base, ffc did not"
        fi
    done
    for f in "$fdir"/*; do
        [ -e "$f" ] || continue
        base=$(basename "$f")
        [ -e "$gdir/$base" ] || echo "NOTE: ffc left $base, gfortran did not"
    done

    return 0
}

# Case 1: READ of an existing default file must return its content and must
# not truncate it.
cat > "$WORK/read_existing.f90" <<'EOF'
program read_existing
    implicit none
    character(20) :: s
    read (unit=10, fmt='(a)') s
    print *, 's=[', trim(s), ']'
end program read_existing
EOF
run_case read_existing "$WORK/read_existing.f90" "printf 'world\n' > fort.10"

# Case 2: READ of an absent default file is an unhandled end-of-file and must
# terminate with the standard error (gfortran: exit 2).
cat > "$WORK/read_absent.f90" <<'EOF'
program read_absent
    implicit none
    character(20) :: s
    read (unit=10, fmt='(a)') s
    print *, 's=[', trim(s), ']'
end program read_absent
EOF
run_case read_absent "$WORK/read_absent.f90" ""

# Case 3: WRITE keeps truncating, so the fix does not swing the other way.
cat > "$WORK/write_truncates.f90" <<'EOF'
program write_truncates
    implicit none
    write (unit=10, fmt='(a)') 'BB'
end program write_truncates
EOF
run_case write_truncates "$WORK/write_truncates.f90" \
    "printf 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n' > fort.10"

# Case 4: an explicitly OPENed unit is untouched by this change.
cat > "$WORK/opened_unit.f90" <<'EOF'
program opened_unit
    implicit none
    integer :: u, ios
    character(20) :: s
    open(newunit=u, file='in.txt', status='old', iostat=ios)
    read(u, '(a)', iostat=ios) s
    print *, 'ios=', ios, 's=[', trim(s), ']'
    close(u)
end program opened_unit
EOF
run_case opened_unit "$WORK/opened_unit.f90" "printf 'hello\n' > in.txt"

# Case 5: IOSTAT= hands the condition back to the program, so the new
# termination must stay out of the way and the run exits 0 like gfortran.
cat > "$WORK/read_absent_iostat.f90" <<'EOF'
program read_absent_iostat
    implicit none
    integer :: ios
    character(20) :: s
    read (unit=10, fmt='(a)', iostat=ios) s
    print *, 'ios=', ios
end program read_absent_iostat
EOF
run_case read_absent_iostat "$WORK/read_absent_iostat.f90" ""

# Case 6: end= label still transfers control instead of terminating; this is
# the pre-existing path the new check has to coexist with.
cat > "$WORK/read_absent_end.f90" <<'EOF'
program read_absent_end
    implicit none
    character(20) :: s
100 continue
    read (unit=10, fmt='(a)', end=200) s
    print *, 's=[', trim(s), ']'
200 continue
    print *, 'reached end'
end program read_absent_end
EOF
run_case read_absent_end "$WORK/read_absent_end.f90" ""

if [ "$fail" -ne 0 ]; then
    echo "default-unit io parity: FAILED"
    exit 1
fi
echo "default-unit io parity: PASS"
exit 0
