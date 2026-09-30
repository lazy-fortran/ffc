#!/usr/bin/env bash
# USE ONLY operator generic-spec spelling parity (#2887).
#
# F2018 19.3.2.3: a user-defined relational operator defined under one
# spelling may be referenced through the other, so a module declaring
# `interface operator(.ne.)` exports the generic spec that
# `use m, only: operator(/=)` imports. ffc compared the literal token, so that
# valid pair was a hard rejection where gfortran accepted it.
#
# The negative half is the point: canonicalising spellings must not become a
# wildcard. Distinct classes, and operators the module never declares, must
# still be rejected. Operands are a derived type because a user-defined
# relational operator over integers would conflict with the intrinsic.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FFC="${FFC:-$ROOT/build/fo/bin/ffc}"
WORK="$(mktemp -d /tmp/ffc-use-only-op.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail=0

# emit <dir> <name> <decl> <want>
emit() {
    local dir="$1" mod="$2" decl="$3" want="$4"
    cat > "$dir/$mod.f90" <<EOF
module $mod
    implicit none
    type wt
        integer :: i
    end type wt
    interface operator($decl)
        module procedure ${mod}_fn
    end interface
contains
    logical function ${mod}_fn(a, b)
        type(wt), intent(in) :: a, b
        ${mod}_fn = a%i > b%i
    end function ${mod}_fn
end module $mod

program user
    use $mod, only: operator($want)
    implicit none
    print *, 'ok'
end program user
EOF
}

# accept_case <decl> <want> <tag>
accept_case() {
    local decl="$1" want="$2" tag="acc_$3" dir rc_g rc_f
    dir="$WORK/$tag"; mkdir -p "$dir"
    emit "$dir" "$tag" "$decl" "$want"
    [ -s "$dir/$tag.f90" ] || { echo "FAIL: $tag fixture missing"; fail=1; return; }
    gfortran -std=f2018 -fsyntax-only "$dir/$tag.f90" >"$dir/g.out" 2>&1; rc_g=$?
    ( cd "$dir" && timeout 60 "$FFC" "$tag.f90" -o "$dir/x.bin" ) >"$dir/f.out" 2>&1; rc_f=$?
    if [ "$rc_g" -ne 0 ]; then
        echo "FAIL: gfortran calls valid pair $decl / $want invalid"
        grep -m2 -E "Error|error" "$dir/g.out"
        fail=1; return
    fi
    if [ "$rc_f" -ne 0 ]; then
        echo "FAIL: valid pair $decl exports $want was rejected by ffc"
        grep -m2 -E "error|Error" "$dir/f.out"
        fail=1; return
    fi
    echo "ok: $decl exports $want"
}

# reject_case <decl> <want> <tag>
reject_case() {
    local decl="$1" want="$2" tag="rej_$3" dir rc_g rc_f
    dir="$WORK/$tag"; mkdir -p "$dir"
    emit "$dir" "$tag" "$decl" "$want"
    [ -s "$dir/$tag.f90" ] || { echo "FAIL: $tag fixture missing"; fail=1; return; }
    gfortran -std=f2018 -fsyntax-only "$dir/$tag.f90" >"$dir/g.out" 2>&1; rc_g=$?
    ( cd "$dir" && timeout 60 "$FFC" "$tag.f90" -o "$dir/x.bin" ) >"$dir/f.out" 2>&1; rc_f=$?
    if [ "$rc_g" -eq 0 ]; then
        echo "FAIL: gfortran accepts a pair this test claims is invalid: $decl / $want"
        fail=1; return
    fi
    if [ "$rc_f" -eq 0 ]; then
        echo "FAIL: distinct operator $want imported from $decl was accepted"
        fail=1; return
    fi
    echo "ok: $decl does not export $want (both reject)"
}

# Relational classes are spelling-interchangeable in both directions.
accept_case '.eq.' '=='   eq_sym
accept_case '=='   '.eq.' sym_eq
accept_case '.ne.' '/='   ne_sym
accept_case '/='   '.ne.' sym_ne
accept_case '.lt.' '<'    lt_sym
accept_case '<'    '.lt.' sym_lt
accept_case '.le.' '<='   le_sym
accept_case '<='   '.le.' sym_le
accept_case '.gt.' '>'    gt_sym
accept_case '>'    '.gt.' sym_gt
accept_case '.ge.' '>='   ge_sym
accept_case '>='   '.ge.' sym_ge
accept_case '.ne.' '.ne.' same_ne
accept_case '=='   '=='    same_eq

# Distinct classes stay distinct: lt and le are different generic specs, and
# eq/ne likewise. These are the pairs gfortran also rejects (19.3.2.3 ties
# the equivalence to the intrinsic relational operators, not across classes).
reject_case '.lt.' '<='   lt_le
reject_case '.gt.' '>='   gt_ge

# An operator the module never declares must still be rejected.
reject_case '.and.' '.or.' and_or
reject_case '.not.' '.and.' not_and

# Real corpus fixture that motivated the fix.
FIX="$ROOT/../fortfront/examples/f90/interface_operator_3_corrected.f90"
if [ -f "$FIX" ]; then
    if timeout 60 "$FFC" "$FIX" -o "$WORK/corpus.bin" >"$WORK/corpus.out" 2>&1; then
        echo "ok: interface_operator_3_corrected.f90 compiles"
    else
        echo "FAIL: corpus fixture still rejected"
        head -4 "$WORK/corpus.out"
        fail=1
    fi
fi

# BIND(C) negative fixture. This cluster was reported as a false reject and is
# not one: ffc already agrees with gfortran to the letter. Pinned so a future
# relaxation of BIND(C) name checking cannot pass silently.
BINDNEG="$ROOT/../fortfront/examples/f90/pr89943_3.f90"
if [ -f "$BINDNEG" ]; then
    gfortran -std=f2018 -fsyntax-only "$BINDNEG" >"$WORK/bindneg.g" 2>&1; g_rc=$?
    timeout 60 "$FFC" "$BINDNEG" -o "$WORK/bindneg.bin" >"$WORK/bindneg.f" 2>&1; f_rc=$?
    if [ "$g_rc" -eq 0 ]; then
        echo "FAIL: gfortran now accepts the BIND(C) mismatch fixture"
        fail=1
    elif [ "$f_rc" -eq 0 ]; then
        echo "FAIL: ffc accepted a genuine BIND(C) name mismatch (runFu/runFoo)"
        fail=1
    elif ! grep -qi "runFu" "$WORK/bindneg.f" || ! grep -qi "runFoo" "$WORK/bindneg.f"; then
        echo "FAIL: BIND(C) rejection did not name the mismatching labels"
        head -3 "$WORK/bindneg.f"
        fail=1
    else
        echo "ok: BIND(C) runFu/runFoo mismatch still rejected, matching gfortran"
    fi
fi

# Positive BIND(C) fixture: the matching-label neighbour must compile.
BINDPOS="$ROOT/../fortfront/examples/f90/submodule_bind_c_name_valid.f90"
if [ -f "$BINDPOS" ]; then
    if timeout 60 "$FFC" "$BINDPOS" -o "$WORK/bindpos.bin" >"$WORK/bindpos.out" 2>&1; then
        echo "ok: matching BIND(C) labels compile"
    else
        echo "FAIL: valid BIND(C) name fixture rejected"
        head -3 "$WORK/bindpos.out"
        fail=1
    fi
fi

[ "$fail" -eq 0 ] && echo "use-only operator spelling: PASS" || echo "use-only operator spelling: FAILED"
exit "$fail"
