#!/usr/bin/env python3
"""Byte-exact gfortran parity for MASK reductions, MERGE, and nested ABS (#766).

Every accepted program must match on at least 20 executions. Binary and output
MD5 hashes make the measured artifacts identifiable. Only DIM/KIND remain named
refusals; unsupported inputs must fail compilation rather than emit wrong output.
Numeric regression rows pin MOD signs, integer rounding, integer(8) output,
overflow wraparound, and real rounding alongside the new expression paths.

Build the compiler first with ``fo exec ffc --help``, then run this script.
FFC may override the compiler command; WORK selects the /var/tmp evidence folder.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = shlex.split(os.environ.get("FFC", "fo exec --no-build ffc"))
RUNS = max(20, int(os.environ.get("RUNS", "20")))
WORK = Path(os.environ.get("WORK", "/var/tmp/ffc-goal/perf/mask"))
REPORT = WORK / "report.tsv"

DECL = ("integer :: a(5)\n  integer :: b(5)\n  real :: r(4)\n  logical :: lm(5)\n  lm = [.true.,.true.,.false.,.true.,.true.]\n"
        "  a = [1,2,3,4,5]\n  b = [10,20,30,40,50]\n"
        "  r = [1.5,-2.5,3.0,-4.0]")

# KIND/DIM remain explicit refusals; silently dropping either changes semantics.
KNOWN_REFUSED = ["dim_refused", "dim_pos_expr_refused", "kind_refused"]

CASES = [
    ("sum_plain", "sum(a)"), ("maxval_plain", "maxval(a)"),
    ("minval_plain", "minval(a)"), ("product_plain", "product(a)"),
    ("count_pos", "count(a>2)"), ("any_pos", "any(a>4)"),
    ("all_pos", "all(a>0)"), ("all_false", "all(a>2)"),
    # Nested ABS preserves the element kind on integer and real arrays.
    ("abs_nested_int", "maxval(abs(a))"),
    ("abs_nested_min", "minval(abs(a))"),
    ("abs_nested_sum", "sum(abs(-1*a))"),
    ("abs_nested_product", "product(abs(a))"),
    ("abs_nested_real", "maxval(abs(r))"),
    ("abs_nested_real_sum", "sum(abs(r))"),
    ("abs_nested_section", "sum(abs(a(2:4)))"),
    ("abs_scalar_int", "abs(-5)"),
    ("abs_scalar_real", "abs(-5.5)"),
    ("sum_mask", "sum(a, mask=a>2)"),
    ("sum_mask_all_true", "sum(a, mask=a>0)"),
    ("sum_mask_none", "sum(a, mask=a>99)"),
    ("maxval_mask", "maxval(a, mask=a<4)"),
    ("maxval_mask_excludes_max", "maxval(a, mask=a<5)"),
    ("maxval_mask_none", "maxval(a, mask=a>99)"),
    ("minval_mask", "minval(a, mask=a<4)"),
    ("minval_mask_excludes_min", "minval(a, mask=a>1)"),
    ("minval_mask_none", "minval(a, mask=a>99)"),
    ("product_mask", "product(a, mask=a<3)"),
    ("product_mask_none", "product(a, mask=a>99)"),
    ("count_kw_mask", "count(mask=a>2)"),
    ("sum_pos_mask", "sum(a, a>2)"),
    ("sum_mask_ident", "sum(a, mask=lm)"),
    ("sum_mask_scalar_true", "sum(a, mask=.true.)"),
    ("sum_mask_scalar_false", "sum(a, mask=.false.)"),
    ("maxval_mask_scalar_false", "maxval(a, mask=.false.)"),
    ("minval_mask_scalar_false", "minval(a, mask=.false.)"),
    ("sum_mask_logical_tree", "sum(a, mask=a>1.and.a<5)"),
    ("sum_mask_logical_tree_spaced", "sum(a, mask=a > 1 .and. a < 5)"),
    ("sum_mask_logical_not", "sum(a, mask=.not.lm)"),
    ("sum_mask_real_threshold", "sum(a, mask=a>2.5)"),
    ("sum_r_mask", "sum(r, mask=r>0)"),
    ("maxval_r_mask", "maxval(r, mask=r>0)"),
    ("maxval_r_mask_none", "maxval(r, mask=r>100.0)"),
    ("minval_r_mask", "minval(r, mask=r<-1.0)"),
    ("minval_r_mask_none", "minval(r, mask=r>100.0)"),
    ("product_r_mask", "product(r, mask=r>0)"),
    ("dim_refused", "sum(a, dim=1)"),
    ("dim_pos_expr_refused", "sum(a, 1+0)"),
    ("kind_refused", "maxval(a, mask=a>2, kind=8)"),
    ("merge_scalar", "merge(1,2,a>2)"),
    ("merge_array", "merge(a,b,a>2)"),
    ("merge_array_logical_mask", "merge(a,b,lm)"),
    ("merge_array_true", "merge(a,b,.true.)"),
    ("merge_array_false", "merge(a,b,.false.)"),
    ("merge_source_broadcast", "merge(a,99,a>2)"),
    ("merge_false_source_array", "merge(99,b,a>2)"),
    ("merge_array_expr", "merge(a+1,b*2,a>2)"),
    ("merge_mask_logical_tree", "merge(a,b,a>1.and.a<5)"),
    ("merge_mask_logical_tree_spaced", "merge(a,b,a > 1 .and. a < 5)"),
    ("merge_mask_not", "merge(a,b,.not.lm)"),
    ("merge_logical_comparison_sources", "merge(a>2,b<30,lm)"),
    ("merge_kw_reorder", "merge(mask=a>2,fsource=b,tsource=a)"),
    ("merge_nested", "merge(merge(a,b,a>2),b,lm)"),
    ("merge_real_array", "merge(r,0.0,r>0)"),
    ("merge_scalar_int_runtime", "__LOCAL__"),
    ("merge_scalar_real_runtime", "__LOCAL__"),
    ("merge_scalar_double_runtime", "__LOCAL__"),
    ("merge_scalar_logical_runtime", "__LOCAL__"),
    ("merge_scalar_kw_runtime", "__LOCAL__"),
    # numeric core - passes today, pinned as regression guards
    ("div_mod_neg", "__LOCAL__"),
    ("trunc_fns", "__LOCAL__"),
    ("int8_huge", "__LOCAL__"),
    ("overflow_wrap", "__LOCAL__"),
    ("real_round", "__LOCAL__"),
    ("mod_sign", "__LOCAL__"),
]

SPECIAL = {
    "merge_scalar_int_runtime": (
        "integer :: i,j\n  logical :: m\n  i=7\n  j=9\n  m=i>j",
        "merge(i,j,m), merge(i,j,.not.m)",
    ),
    "merge_scalar_real_runtime": (
        "real :: x,y\n  logical :: m\n  x=1.5\n  y=-2.5\n  m=x>y",
        "merge(x,y,m), merge(x,y,.not.m)",
    ),
    "merge_scalar_double_runtime": (
        "real(8) :: x,y\n  logical :: m\n  x=1.5d0\n  y=-2.5d0\n  m=x>y",
        "merge(x,y,m), merge(x,y,.not.m)",
    ),
    "merge_scalar_logical_runtime": (
        "logical :: x,y,m\n  x=.true.\n  y=.false.\n  m=.false.",
        "merge(x,y,m), merge(x,y,.not.m)",
    ),
    "merge_scalar_kw_runtime": (
        "integer :: i,j\n  logical :: m\n  i=7\n  j=9\n  m=i>j",
        "merge(mask=m,fsource=j,tsource=i)",
    ),
    "div_mod_neg": ("integer :: x,y\n  x=-7\n  y=2", "x/y, mod(x,y)"),
    "trunc_fns": ("real :: q\n  q=-2.7", "int(q), nint(q), floor(q), ceiling(q)"),
    "int8_huge": ("integer(8) :: k\n  k=9223372036854775807_8", "k"),
    "overflow_wrap": ("integer :: i\n  i=2147483647", "i+1"),
    "real_round": ("real :: pr\n  pr=1.0/3.0", "pr, pr*3.0"),
    "mod_sign": ("integer :: m,n\n  m=7\n  n=-2", "m/n, mod(m,n)"),
}


def run(c):
    p = subprocess.run(c, cwd=ROOT, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    lines, runs, match, refused = [], 0, 0, 0
    for name, expr in CASES:
        decl, body = DECL, expr
        if expr == "__LOCAL__":
            decl, body = SPECIAL[name]
        src = WORK / f"{name}.f90"
        src.write_text(f"program prog\n  implicit none\n  {decl}\n  print *, {body}\n"
                       "end program prog\n")
        grc, _ = run(["gfortran", str(src), "-o", str(WORK / f"{name}_r")])
        if grc != 0 and name in KNOWN_REFUSED:
            # Invalid on this gfortran (e.g. KIND= predates it on maxval);
            # ffc refusing the same input is agreement, not a bad row.
            frc0, ferr0 = run([*FFC, str(src), "-o", str(WORK / f"{name}_f")])
            runs += 1
            if frc0 != 0:
                refused += 1
                lines.append(f"{name}\tKNOWN_REFUSED\tboth_refuse")
            else:
                lines.append(f"{name}\tREGRESSION\tffc_accepts_invalid")
            continue
        if grc != 0:
            lines.append(f"{name}\tBAD_CASE\tgfortran_refuses")
            continue
        frc, ferr = run([*FFC, str(src), "-o", str(WORK / f"{name}_f")])
        runs += 1
        if frc != 0:
            if name in KNOWN_REFUSED:
                refused += 1
                lines.append(f"{name}\tKNOWN_REFUSED\t{ferr.strip()[:40]}")
            else:
                lines.append(f"{name}\tUNEXPECTED_REFUSE\t{ferr.strip()[:40]}")
            continue
        grc, g = run([str(WORK / f"{name}_r")])
        gm = hashlib.md5(g.encode()).hexdigest()
        binary_md5 = hashlib.md5((WORK / f"{name}_f").read_bytes()).hexdigest()
        for repetition in range(RUNS):
            frc, f = run([str(WORK / f"{name}_f")])
            fm = hashlib.md5(f.encode()).hexdigest()
            if grc != 0 or frc != 0 or gm != fm:
                lines.append(f"{name}\tREGRESSION\trun={repetition + 1}\t"
                             f"g={g.strip()!r}\tffc={f.strip()!r}")
                break
        else:
            match += 1
            lines.append(f"{name}\tMATCH\toutput_md5={gm}\t"
                         f"binary_md5={binary_md5}\trepetitions={RUNS}")
    REPORT.write_text("\n".join(lines) + "\n")
    bad = [l for l in lines if l.split("\t")[1] in ("REGRESSION", "UNEXPECTED_REFUSE",
                                                      "BAD_CASE")]
    print(f"runs={runs} match={match} known_refused={refused} unexpected={len(bad)}")
    print(f"report={REPORT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
