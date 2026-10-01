#!/usr/bin/env python3
"""Byte-exact parity for logical values in list-directed output (#761).

Before the fix, `print *, 1<2` was **refused** - "unsupported integer operator:
direct LIRIC session supports +, -, *, and /" - because a comparison reached the
integer arithmetic lowerer. `print "(L1)", 1<2` already printed `T`, so the
comparison lowering and the logical printer were both fine; the list-directed value
path simply had no route for an *intrinsic* logical-valued operator, while the
user-overloaded branch right next to it already called lower_print_logical_value.
The fix mirrors that branch.

An intermediate state of this change printed `1` and `0` instead of `T` and `F`:
routing the value without routing the printer converts an honest refusal into a
wrong answer. These rows pin the T/F spelling so that regression cannot return
quietly. Arithmetic in the same statement is checked in the same rows, because the
new branch sits in front of the arithmetic lowerer and must not disturb it.

Rows gfortran refuses are REF_FAIL (reported, skipped, never a pass).

Run:  python3 tools/test_logical_print_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/logprint")
REPORT = Path("/var/tmp/ffc-goal/perf/logprint/report.tsv")

# Known gaps kept visible by name, never deleted, so the count can only fall.
#  - dotted_lt/dotted_ge: `print *, 1.lt.2` prints `   1.00000000`, the left
#    literal, where gfortran prints `T`. A WRONG ANSWER, not a refusal: a real
#    literal may end in `.`, so `1.lt.2` mis-splits into `1.` / `lt` / `.2`
#    (#763). Symbolic operators (`<`, `>=`) are unaffected and match above.
#  - not_true/not_false: `.not.` is unary, and the #761 route added here covers
#    binary ops only. `print "(L1)", .not.(1>2)` already prints `T`, so the
#    logical machinery is fine and only the list-directed unary route is missing
#    (tracked as a follow-up step on #761).
KNOWN_GAP = ["dotted_lt", "dotted_ge", "not_true", "not_false"]

DECL = 'integer :: i, j\n  real :: x, y\n  character(len=5) :: s, t\n  s="ab"\n  t="ab"'

CASES = [
    ("lt_true", "print *, 1<2"),
    ("lt_false", "print *, 2<1"),
    ("gt_true", "print *, 3>2"),
    ("gt_false", "print *, 2>3"),
    ("eq_true", "print *, 2==2"),
    ("eq_false", "print *, 2==3"),
    ("ne_true", "print *, 1/=2"),
    ("ne_false", "print *, 2/=2"),
    ("le_eq", "print *, 2<=2"),
    ("le_less", "print *, 1<=2"),
    ("ge_eq", "print *, 2>=2"),
    ("ge_false", "print *, 1>=2"),
    ("dotted_lt", "print *, 1.lt.2"),
    ("dotted_ge", "print *, 3.ge.3"),
    ("real_gt", "print *, 1.5>1.0"),
    ("real_lt", "print *, 1.5<2.0"),
    ("real_eq", "print *, 2.0==2.0"),
    ("char_eq", 'print *, s=="ab"'),
    ("char_ne", 'print *, s/="xy"'),
    ("char_lt", 'print *, "abc"<"abd"'),
    ("char_ge", 'print *, s>=t'),
    ("and_true", "print *, .true..and..true."),
    ("and_false", "print *, (.true.).and.(.false.)"),
    ("or_true", "print *, (.true.).or.(.false.)"),
    ("or_false", "print *, .false..or..false."),
    ("not_true", "print *, .not.(1>2)"),
    ("not_false", "print *, .not.(1<2)"),
    ("eqv", "print *, (.true.).eqv.(.false.)"),
    ("neqv", "print *, (.true.).neqv.(.false.)"),
    ("var_cmp", "i=5\n    j=3\n    print *, i>j"),
    ("var_cmp_eq", "i=4\n    j=4\n    print *, i==j"),
    ("var_arith_cmp", "i=2\n    j=3\n    print *, i+j>4"),
    ("cmp_then_arith", "print *, 1<2, 3+4"),
    ("mixed_multi", 'print *, 1<2, 3>4, "x"'),
    ("arith_unchanged", "print *, 2+3, 7-1, 3*4, 8/2, 2**3"),
    ("logical_var", "print *, s==t, len(s)"),
    ("two_cmp", "print *, 1<2, 2<1"),
    ("cmp_neg_int", "print *, -3<0"),
]


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    if not FFC.exists():
        print(f"SKIP: ffc not built at {FFC}", file=sys.stderr)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    runs = match = refused = mismatch = ref_fail = 0
    for name, stmt in CASES:
        gap = name in KNOWN_GAP
        src = WORK / f"{name}.f90"
        src.write_text("program p\n  implicit none\n  " + DECL + "\n    " + stmt +
                       "\nend program p\n")
        rc_g, _, gerr = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
        if rc_g != 0:
            ref_fail += 1
            lines.append(f"{name}\tREF_FAIL\t{gerr.strip()[:40]}")
            continue
        rc_f, _, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
        if rc_f != 0 and gap:
            lines.append(f"{name}\tKNOWN_GAP\tREFUSED")
            continue
        if rc_f != 0:
            refused += 1
            lines.append(f"{name}\tREFUSED\t{ferr.strip()[:40]}")
            continue
        rr = run([str(WORK / f"{name}_ref")])
        fr = run([str(WORK / f"{name}_ffc")])
        rmd5 = hashlib.md5(rr[1].encode()).hexdigest()
        fmd5 = hashlib.md5(fr[1].encode()).hexdigest()
        runs += 1
        if gap and rmd5 != fmd5:
            lines.append(f"{name}\tKNOWN_GAP\tWRONG_OUTPUT\tffc={fr[1]!r}")
            continue
        if rmd5 == fmd5:
            match += 1
            lines.append(f"{name}\tMATCH\tref={rmd5[:12]}\tffc={fmd5[:12]}")
        else:
            mismatch += 1
            lines.append(
                f"{name}\tMISMATCH\tref={rmd5[:12]}:{rr[1]!r}\tffc={fmd5[:12]}:{fr[1]!r}")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} refused={refused} mismatch={mismatch} "
          f"ref_fail={ref_fail}")
    print(f"report={REPORT}")
    return 1 if (mismatch or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
