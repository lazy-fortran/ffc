#!/usr/bin/env python3
"""Guard the derived-type access/assignment surface (PLAN W1 #339/#348 adjacency).

Probed against gfortran because the descriptor-retirement clusters (#339, #348)
edit the machinery these shapes travel through. Derived types currently match
byte-exactly: component read/write, nested `o%a%v`, character components with
`trim`, and a derived value passed to a function by host association.

No defect is claimed here - this is a pinned baseline. Its purpose is to fail when
a descriptor change breaks a shape that works today, which is how #339 slice D was
caught before deleting anything: the audit found two legitimate descriptor producers
where the plan expected legacy writes to remove, and a guard is what keeps that
distinction from being re-litigated by a future refactor.

`KNOWN_GAP` rows are valid Fortran refused today, kept visible by name:
`new_line("A")` (unsupported character expression) and a `select type` on a class
selector, which stays at its visible refusal because cross-unit `class(T)` dispatch
is out of scope per the plan boundary (#417).

Run:  python3 tools/test_derived_type_access_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/dtype")
REPORT = Path("/var/tmp/ffc-goal/perf/dtype/report.tsv")

DT = ("  type :: pt\n    integer :: x\n    integer :: y\n  end type pt\n")
NESTED = (
    "  type :: inner\n    integer :: v\n  end type inner\n"
    "  type :: outer\n    type(inner) :: a\n    type(inner) :: b\n  end type outer\n"
)
REC = ("  type :: rec\n    character(len=8) :: name\n    integer :: n\n  end type rec\n")

# (name, preamble, declarations, statements)
CASES = [
    ("rd_x", DT, "type(pt) :: q\n  q%x=3\n  q%y=4", "print *, q%x"),
    ("rd_y", DT, "type(pt) :: q\n  q%x=3\n  q%y=4", "print *, q%y"),
    ("sum_xy", DT, "type(pt) :: q\n  q%x=3\n  q%y=4", "print *, q%x+q%y"),
    ("prod_xy", DT, "type(pt) :: q\n  q%x=3\n  q%y=4", "print *, q%x*q%y"),
    ("diff_xy", DT, "type(pt) :: q\n  q%x=9\n  q%y=4", "print *, q%x-q%y"),
    ("assign_then_read", DT, "type(pt) :: q\n  q%x=1", "q%y=2\n  print *, q%x, q%y"),
    ("copy_comp", DT, "type(pt) :: q, r\n  q%x=5\n  q%y=6", "r%x=q%x\n  print *, r%x"),
    ("two_objs", DT, "type(pt) :: a, b\n  a%x=1\n  b%x=2", "print *, a%x, b%x"),
    ("nested_a", NESTED, "type(outer) :: o\n  o%a%v=10", "print *, o%a%v"),
    ("nested_sum", NESTED, "type(outer) :: o\n  o%a%v=10\n  o%b%v=20",
     "print *, o%a%v+o%b%v"),
    ("nested_diff", NESTED, "type(outer) :: o\n  o%a%v=30\n  o%b%v=8",
     "print *, o%a%v-o%b%v"),
    ("nested_assign", NESTED, "type(outer) :: o",
     "o%a%v=1\n  o%b%v=2\n  print *, o%a%v, o%b%v"),
    ("char_trim", REC, 'type(rec) :: r\n  r%name="hello"\n  r%n=7',
     "print *, trim(r%name), r%n"),
    ("char_len", REC, 'type(rec) :: r\n  r%name="abc"',
     "print *, len(r%name), len_trim(r%name)"),
    ("char_concat", REC, 'type(rec) :: r\n  r%name="ab"',
     'print *, trim(r%name)//"Z"'),
    ("int_field_zero", REC, 'type(rec) :: r\n  r%name="x"\n  r%n=0', "print *, r%n"),
    ("comp_neg", DT, "type(pt) :: q\n  q%x=-5\n  q%y=-2", "print *, q%x, q%y, q%x+q%y"),
    ("comp_double_assign", DT, "type(pt) :: q",
     "q%x=1\n  q%x=q%x+10\n  print *, q%x"),
    ("self_ref", DT, "type(pt) :: q\n  q%x=2", "q%x=q%x*q%x\n  print *, q%x"),
    ("dt_func_arg", DT, "type(pt) :: q\n  q%x=5", "print *, f(q)"),
    ("dt_func_arg2", DT, "type(pt) :: q\n  q%x=7\n  q%y=3", "print *, f(q)"),
    ("dt_sub_call", DT, "type(pt) :: q\n  q%x=4", "call g(q)\n  print *, q%x"),
    ("nested_char", REC, 'type(rec) :: r1, r2\n  r1%name="aa"\n  r2%name="bb"',
     'print *, trim(r1%name), trim(r2%name)'),
]

# Valid Fortran refused today - kept visible, not failed.
# (A `select type` row was removed: `select type (z => q)` on a concrete type is
# not valid Fortran - the selector must be polymorphic - so it tested nothing, and
# polymorphic dispatch is out of scope here per the #417 boundary.)
KNOWN_GAP = ["newline_intr"]
GAP = [
    ("newline_intr", DT, 'type(pt) :: q', 'print *, "A"//new_line("A")//"B"'),
]

HELPERS = (
    "contains\n"
    "  integer function f(p)\n    type(pt) :: p\n    f = p%x*2\n"
    "  end function f\n"
    "  subroutine g(p)\n    type(pt) :: p\n    p%x = p%x+1\n"
    "  end subroutine g\n"
)


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def check(name: str, pre: str, decl: str, stmt: str, lines: list[str]) -> tuple[int, int, int]:
    src = WORK / f"{name}.f90"
    needs_helpers = "f(q)" in stmt or "g(q)" in stmt
    src.write_text(
        "program p\n    implicit none\n" + pre + "  " + decl + "\n  " + stmt + "\n"
        + (HELPERS if needs_helpers else "") + "end program p\n"
    )
    rc_g, _, gerr = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
    if rc_g != 0:
        lines.append(f"{name}\tREF_FAIL\t{gerr.strip()[:36]}")
        return 0, 0, 0
    rc_f, _, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
    if rc_f != 0:
        tag = "REFUSED" if "unsupported" in ferr else "OTHER_ERROR"
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
    for name, pre, decl, stmt in CASES + GAP:
        r, m, f = check(name, pre, decl, stmt, lines)
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
