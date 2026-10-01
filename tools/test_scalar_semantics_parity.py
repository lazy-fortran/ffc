#!/usr/bin/env python3
"""Guard scalar semantics that already match gfortran (#761 adjacency).

Found while probing, not from the plan: `print *, a=="ab"` with a character `a`
is refused ("integer expression used non-integer identifier: a") while
`if (a=="ab")` lowers correctly. #761 records the root cause - a missing
VALUE_CHARACTER branch beside the existing VALUE_LOGICAL one in
`lower_integer_expression`.

This oracle pins the half that is CORRECT today, because the #761 fix edits
exactly that dispatcher and must not regress it:

- integer division and modulo across all four sign combinations (the classic
  trap - Fortran truncates toward zero, and a floor-division implementation
  looks right until a negative operand appears);
- integer and real exponentiation;
- `index`, `scan`-style searches, `len_trim`, `adjustl`;
- character comparison in **conditional** context, which works, and stands as the
  executable proof that the comparison lowering itself is fine and only the
  print-item route is missing.

Rows gfortran refuses are `REF_FAIL` (reported, skipped, never a pass). Character
comparisons **as print items** are listed in `KNOWN_GAP` and reported, so the gap
stays visible without turning the guard red; deleting them would hide #761.

Run:  python3 tools/test_scalar_semantics_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/scalsem")
REPORT = Path("/var/tmp/ffc-goal/perf/scalsem/report.tsv")

DECL = {
    "int": "integer :: i, j\n  integer :: ia(4)",
    "char": 'character(len=5) :: a, b\n  a="ab"\n  b="ab"',
    "real": "real :: x",
    "i8": "integer(8) :: big",
}

# (name, decl-key, statements)
CASES = [
    ("div_pos", "int", "print *, 7/2"),
    ("div_neg_num", "int", "print *, -7/2"),
    ("div_neg_den", "int", "print *, 7/(-2)"),
    ("div_both_neg", "int", "print *, (-7)/(-2)"),
    ("mod_pos", "int", "print *, mod(7,2)"),
    ("mod_neg_num", "int", "print *, mod(-7,2)"),
    ("mod_neg_den", "int", "print *, mod(7,-2)"),
    ("mod_both_neg", "int", "print *, mod(-7,-2)"),
    ("div_trunc_toward_zero", "int", "print *, (-9)/2, mod(-9,2)"),
    ("pow_int_pos", "int", "print *, 2**10"),
    ("pow_int_neg_base", "int", "print *, (-2)**3"),
    ("pow_zero", "int", "print *, 2**0"),
    ("pow_real", "real", "print *, 2.0**0.5"),
    ("pow_neg_exp", "real", "print *, 2.0**(-1)"),
    ("index_found", "char", 'print *, index("hello world","world")'),
    ("index_absent", "char", 'print *, index("hello","z")'),
    ("len_trim_blank", "char", 'print *, len_trim("  ab  ")'),
    ("adjustl_pad", "char", 'print *, adjustl("  ab")'),
    ("adjustr_pad", "char", 'print *, adjustr("ab  ")'),
    ("cmp_if_eq", "char", 'if (a=="ab") print *, "yes"'),
    ("cmp_if_ne", "char", 'if (a/="xy") print *, "no"'),
    ("cmp_if_lt", "char", 'if ("abc"<"abd") print *, "lt"'),
    ("cmp_if_var", "char", "if (a==b) print *, \"same\""),
    ("char_assign_len", "char", "print *, len(a), len_trim(a)"),
    ("char_slice_print", "char", "print *, a(1:2)"),
    ("int_var_arith", "int", "i=5\n  j=3\n  print *, i+j, i-j, i*j"),
    ("array_elem_read", "int", "ia=[10,20,30,40]\n  print *, ia(3)"),
]

# Valid Fortran refused today: #761. NOT character-specific - any comparison or
# logical in list-directed print position. Kept here by name so the count of known
# refusals is visible and can only fall when the route is added.
KNOWN_GAP = [
    "print_cmp_eq", "print_cmp_ne", "print_cmp_var",
    "print_int_lt", "print_int_eq", "print_int_ne",
    "print_real_gt", "print_logical_and",
]

GAP = [
    ("print_cmp_eq", "char", 'print *, a=="ab"'),
    ("print_cmp_ne", "char", 'print *, a/="xy"'),
    ("print_cmp_var", "char", "print *, a==b"),
    ("print_int_lt", "int", "print *, 1<2"),
    ("print_int_eq", "int", "print *, 2==2"),
    ("print_int_ne", "int", "print *, 1/=2"),
    ("print_real_gt", "real", "print *, 1.5>1.0"),
    ("print_logical_and", "int", "print *, (.true.).and.(.false.)"),
    # Conditional forms of the same expressions - these MUST keep working, and are
    # the executable proof that the comparison lowering is fine (#761 is routing).
    ("if_int_lt", "int", 'if (1<2) print *, "ok"'),
    ("if_real_gt", "real", 'if (1.5>1.0) print *, "big"'),
    ("if_logical_and", "int", 'if (.true..and..false.) print *, "no"'),
    ("if_logical_or", "int", 'if (.true..or..false.) print *, "yes"'),
]


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def check(name: str, key: str, stmt: str, lines: list[str]) -> tuple[int, int, int]:
    src = WORK / f"{name}.f90"
    src.write_text(
        "program p\n  implicit none\n  " + DECL[key] + "\n  " + stmt + "\nend program p\n"
    )
    rc_g, _, gerr = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
    if rc_g != 0:
        lines.append(f"{name}\tREF_FAIL\t{gerr.strip()[:36]}")
        return 0, 0, 0
    rc_f, _, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
    if rc_f != 0:
        tag = "REFUSED" if ("unsupported" in ferr or "non-integer" in ferr) else "OTHER_ERROR"
        if name in KNOWN_GAP:
            lines.append(f"{name}\tKNOWN_GAP\t{tag}")
            return 0, 0, 0
        lines.append(f"{name}\tREGRESSED_{tag}\t{ferr.strip()[:36]}")
        return 1, 0, 1
    rr = run([str(WORK / f"{name}_ref")])
    fr = run([str(WORK / f"{name}_ffc")])
    rmd5 = hashlib.md5(rr[1].encode()).hexdigest()
    fmd5 = hashlib.md5(fr[1].encode()).hexdigest()
    if rmd5 == fmd5:
        lines.append(f"{name}\tMATCH\t{rmd5}")
        return 1, 1, 0
    lines.append(f"{name}\tMISMATCH\tref={rmd5[:12]}:{rr[1]!r}\tffc={fmd5[:12]}:{fr[1]!r}")
    return 1, 0, 1


def main() -> int:
    if not FFC.exists():
        print(f"SKIP: ffc not built at {FFC}", file=sys.stderr)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    runs = match = fail = 0
    for name, key, stmt in CASES + GAP:
        r, m, f = check(name, key, stmt, lines)
        runs += r
        match += m
        fail += f
    gap = sum(1 for l in lines if l.split("\t")[1] == "KNOWN_GAP")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} fail={fail} known_gap={gap}")
    print(f"report={REPORT}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
