#!/usr/bin/env python3
"""Guard the allocatable / intent / optional runtime surface (#762 adjacency).

Probed against gfortran because #339/#348 retire parallel shape ledgers in favour
of the canonical descriptor, and this is the surface those edits touch.

Everything in CASES matches gfortran byte-exactly today - allocatable scalar and
rank-1 array, allocation, deallocation, reallocation to a different extent,
element and whole-array reads, `size`/`size(a,dim)`, `allocated`, `intent(out)`
definition, `intent(inout)` accumulation, `optional` with `present` on both
branches, and a rank-1 lower bound of 1 via `lbound`/`ubound`.

KNOWN_GAP rows are valid Fortran refused today and stay listed **by name** so the
count of known refusals is visible and can only fall:
  - `lb_offset_*` -> ffc#762: `allocate(a(2:4))` is refused although
    docs/SUPPORT_CONTRACT.md promises rank-1 runtime allocate plus bounds access,
    and the refusal message blames "integer expressions", naming the wrong
    construct. Element address with an offset base is `base + (i-lbound)*stride`,
    which is precisely where a parallel shape ledger and a real descriptor disagree.
  - `char_alloc_scalar` -> scalar `character, allocatable` allocate unsupported.
  - `alloc_2d_reshape` -> rank-2 whole-array assignment from an expression.

Rows gfortran itself refuses are `REF_FAIL` (reported, skipped, never a pass).

Run:  python3 tools/test_allocatable_intent_optional_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/alio")
REPORT = Path("/var/tmp/ffc-goal/perf/alio/report.tsv")

A1 = "integer, allocatable :: a(:)"
HELP_SUB = (
    "contains\n"
    "  subroutine set(z)\n    integer, intent(out) :: z\n    z=42\n"
    "  end subroutine set\n"
    "  subroutine add(z,k)\n    integer, intent(inout) :: z\n"
    "    integer, intent(in) :: k\n    z=z+k\n"
    "  end subroutine add\n"
    "  subroutine sub(x)\n    integer, optional :: x\n"
    "    if (present(x)) then\n      print *, \"got\", x\n"
    "    else\n      print *, \"none\"\n    end if\n"
    "  end subroutine sub\n"
)

# (name, declarations, statements, needs_helpers)
CASES = [
    ("alloc_scalar", "integer, allocatable :: s\n  allocate(s)\n  s=7",
     "print *, s", False),
    ("alloc_scalar_allocated", "integer, allocatable :: s\n  allocate(s)\n  s=7",
     "print *, allocated(s)", False),
    ("alloc_dealloc", "integer, allocatable :: s\n  allocate(s)\n  s=1",
     "deallocate(s)\n  print *, allocated(s)", False),
    ("a_elem1", A1 + "\n  allocate(a(3))\n  a=[10,20,30]", "print *, a(1)", False),
    ("a_elem2", A1 + "\n  allocate(a(3))\n  a=[10,20,30]", "print *, a(2)", False),
    ("a_elem3", A1 + "\n  allocate(a(3))\n  a=[10,20,30]", "print *, a(3)", False),
    ("a_size", A1 + "\n  allocate(a(3))\n  a=[10,20,30]", "print *, size(a)", False),
    ("a_size_dim", A1 + "\n  allocate(a(4))\n  a=[1,2,3,4]",
     "print *, size(a,1)", False),
    ("a_sum", A1 + "\n  allocate(a(3))\n  a=[10,20,30]", "print *, sum(a)", False),
    ("a_lbound1", A1 + "\n  allocate(a(3))\n  a=[1,2,3]",
     "print *, lbound(a,1), ubound(a,1)", False),
    ("a_whole_read", A1 + "\n  allocate(a(3))\n  a=[4,5,6]", "print *, a", False),
    ("a_assign_scalar", A1 + "\n  allocate(a(3))", "a=9\n  print *, a", False),
    ("a_elementwise", A1 + "\n  allocate(a(3))\n  a=[1,2,3]",
     "print *, a(1)+a(2)+a(3)", False),
    ("a_neg", A1 + "\n  allocate(a(3))\n  a=[-1,-2,-3]", "print *, sum(a)", False),
    ("realloc_grow", A1 + "\n  allocate(a(2))\n  a=[1,2]\n  deallocate(a)\n"
     "  allocate(a(4))\n  a=[5,6,7,8]", "print *, size(a), sum(a)", False),
    ("realloc_shrink", A1 + "\n  allocate(a(4))\n  a=[1,2,3,4]\n  deallocate(a)\n"
     "  allocate(a(2))\n  a=[7,8]", "print *, size(a), sum(a)", False),
    ("intent_out", "integer :: v", "call set(v)\n  print *, v", True),
    ("intent_out_twice", "integer :: v",
     "call set(v)\n  call set(v)\n  print *, v", True),
    ("intent_inout", "integer :: v\n  v=1", "call add(v,4)\n  print *, v", True),
    ("intent_inout_accum", "integer :: v\n  v=10",
     "call add(v,1)\n  call add(v,2)\n  print *, v", True),
    ("opt_present", "", "call sub(1)", True),
    ("opt_absent", "", "call sub()", True),
    ("opt_both", "", "call sub(3)\n  call sub()", True),
    ("opt_var", "integer :: k\n  k=5", "call sub(k)\n  call sub()", True),
]

KNOWN_GAP = ["lb_offset_read", "lb_offset_lbound", "lb_offset_size",
             "char_alloc_scalar", "alloc_2d_reshape"]
GAP = [
    ("lb_offset_read", A1 + "\n  allocate(a(2:4))\n  a=[10,20,30]",
     "print *, a(2), a(4)", False),
    ("lb_offset_lbound", A1 + "\n  allocate(a(2:4))\n  a=[10,20,30]",
     "print *, lbound(a,1), ubound(a,1)", False),
    ("lb_offset_size", A1 + "\n  allocate(a(2:4))\n  a=[10,20,30]",
     "print *, size(a)", False),
    ("char_alloc_scalar", 'character(len=10), allocatable :: c\n  allocate(c)\n'
     '  c="hello"', "print *, trim(c), len(c)", False),
    ("alloc_2d_reshape", "integer, allocatable :: m(:,:)\n  allocate(m(2,3))\n"
     "  m=reshape([1,2,3,4,5,6],[2,3])", "print *, m(1,1), m(2,3), sum(m)", False),
]


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def check(name: str, decl: str, stmt: str, helpers: bool,
          lines: list[str]) -> tuple[int, int, int]:
    src = WORK / f"{name}.f90"
    src.write_text(
        "program p\n  implicit none\n  " + decl + "\n  " + stmt + "\n"
        + (HELP_SUB if helpers else "") + "end program p\n"
    )
    rc_g, _, gerr = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
    if rc_g != 0:
        lines.append(f"{name}\tREF_FAIL\t{gerr.strip()[:36]}")
        return 0, 0, 0
    rc_f, _, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
    if rc_f != 0:
        tag = "REFUSED" if ("unsupported" in ferr or "only supports" in ferr
                            or "allocatable" in ferr) else "OTHER_ERROR"
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
    for name, decl, stmt, helpers in CASES + GAP:
        r, m, f = check(name, decl, stmt, helpers, lines)
        runs += r
        match += m
        fail += f
    gap = sum(1 for l in lines if l.split("\t")[1] == "KNOWN_GAP")
    refl = sum(1 for l in lines if l.split("\t")[1] == "REF_FAIL")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} fail={fail} known_gap={gap} ref_fail={refl}")
    print(f"report={REPORT}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
