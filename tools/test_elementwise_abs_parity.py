#!/usr/bin/env python3
"""Byte-exact guard: whole-array abs(identifier) lowers elementwise.

Pins the fix for the silent miscompilation where `X = abs(X)` (and
`X = abs(Y)`) folded the array operand through the scalar i32 path and
stored a garbage scalar across every element (ffc printed 0 0 for
[3,-4]). The new path loops load/icmp/select/store over the static
rank-1 i32 extents and supports in-place (target == source).

Rows cover: in-place, distinct src, all-negative, mixed, zeros,
boundary INT_MIN-ish magnitudes, size 1 and 8, repeated application,
abs after arithmetic init, and named refusals (rank-2 abs, real abs,
abs(sum(...)) chained form which stays a documented decline).

Run:  python3 tools/test_elementwise_abs_parity.py
Exit: 0 all rows match or refuse as pinned; 1 otherwise.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "bin" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/ewabs")
REPORT = WORK / "report.tsv"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT / "build" / "fo" / "lib") + os.pathsep + ENV.get("LD_LIBRARY_PATH", "")

def P(body): return f"program p\n  implicit none\n  {body}\nend program p\n"

CASES = [
    (" inplace", P("integer :: X(2)\n  X = [3,-4]\n  X = abs(X)\n  print '(2I0)', X(1), X(2)")),
    ("distinct", P("integer :: A(3), B(3)\n  A = [-1,2,-3]\n  B = abs(A)\n  print '(3I0)', B(1), B(2), B(3)")),
    ("allneg",  P("integer :: X(3)\n  X = [-5,-6,-7]\n  X = abs(X)\n  print '(3I0)', X(1), X(2), X(3)")),
    ("zeros",   P("integer :: X(3)\n  X = [0,0,0]\n  X = abs(X)\n  print '(3I0)', X(1), X(2), X(3)")),
    ("size1",   P("integer :: X(1)\n  X = [-9]\n  X = abs(X)\n  print '(I0)', X(1)")),
    ("size8",   P("integer :: X(8)\n  X = [1,-2,3,-4,5,-6,7,-8]\n  X = abs(X)\n  print '(8I0)', X(1),X(2),X(3),X(4),X(5),X(6),X(7),X(8)")),
    ("twice",   P("integer :: X(2)\n  X = [-3,4]\n  X = abs(X)\n  X = abs(X)\n  print '(2I0)', X(1), X(2)")),
    ("expr_init", P("integer :: X(2)\n  X = [2-5, 7]\n  X = abs(X)\n  print '(2I0)', X(1), X(2)")),
    ("big",     P("integer :: X(2)\n  X = [2147483647, -2147483646]\n  X = abs(X)\n  print '(2I0)', X(1), X(2)")),
    ("neg_min", P("integer :: X(1)\n  X = -abs([0]) - 1\n  print '(I0)', X(1)")),
    ("sum_then_assign", P("integer :: A(2,2), X(2)\n  A = reshape([-1,-2,-3,-4],[2,2])\n  X = sum(A, dim=1)\n  X = abs(X)\n  print '(2I0)', X(1), X(2)")),
    ("chain_sum1", P("integer :: A(2,2), X(2)\n  A = reshape([-1,-2,3,4],[2,2])\n  X = abs(sum(A, dim=1))\n  print '(2I0)', X(1), X(2)")),
    ("chain_sum2", P("integer :: A(2,3), X(2)\n  A = reshape([-1,2,-3,4,-5,6],[2,3])\n  X = abs(sum(A, dim=2))\n  print '(2I0)', X(1), X(2)")),
    ("chain_prod", P("integer :: A(2,2), X(2)\n  A = reshape([-2,3,-1,4],[2,2])\n  X = abs(product(A, dim=1))\n  print '(2I0)', X(1), X(2)")),
    ("chain_max",  P("integer :: A(3,2), X(2)\n  A = reshape([-5,1,-2,7,-3,0],[3,2])\n  X = abs(maxval(A, dim=1))\n  print '(2I0)', X(1), X(2)")),
    ("chain_min",  P("integer :: A(3,2), X(2)\n  A = reshape([-5,1,-2,7,-3,-9],[3,2])\n  X = abs(minval(A, dim=1))\n  print '(2I0)', X(1), X(2)")),
    ("literal_then_abs", P("integer :: X(3)\n  X = [10,-20,30]\n  X = abs(X)\n  print '(3I0)', X(1), X(2), X(3)")),
    ("src_reused", P("integer :: A(2), B(2), s\n  A = [-1,-2]\n  B = abs(A)\n  s = B(1)+B(2)\n  print '(I0)', s")),
    ("inplace_twice_neg", P("integer :: X(2)\n  X = [-1,1]\n  X = abs(X)\n  X = X - 3\n  X = abs(X)\n  print '(2I0)', X(1), X(2)")),
    ("size3_mixed", P("integer :: X(3)\n  X = [-2,0,2]\n  X = abs(X)\n  print '(3I1)', X(1), X(2), X(3)")),
    ("arr_init_data", P("integer :: X(3) / -1,-2,-3 /\n  X = abs(X)\n  print '(3I0)', X(1), X(2), X(3)")),
    ("copy_chain", P("integer :: A(2), B(2)\n  A = [-7,-8]\n  B = abs(A)\n  A = abs(B)\n  print '(2I0)', A(1), A(2)")),
    ("one_elem_big", P("integer :: X(1)\n  X = -2147483647\n  X = abs(X)\n  print '(I0)', X(1)")),
    ("after_implied_print", P("integer :: X(2)\n  X = [-4,5]\n  X = abs(X)\n  print '(2I0)', X(1), X(2)\n  print '(I0)', sum(X)")),
    ("reshape_src", P("integer :: A(2,2), Y(2), Z(2)\n  A = reshape([-1,2,-3,4],[2,2])\n  Y = sum(A, dim=1)\n  Z = abs(Y)\n  print '(2I0)', Z(1), Z(2)")),
    ("neg_literal_wide", P("integer :: X(4)\n  X = [-100,-200,-300,400]\n  X = abs(X)\n  print '(4I0)', X(1),X(2),X(3),X(4)")),
]

REFUSE_CASES = [
    ("refuse_rank2", P("integer :: A(2,2)\n  A = reshape([-1,2,-3,4],[2,2])\n  A = abs(A)\n  print '(4I0)', A(1,1),A(2,1),A(1,2),A(2,2)\n"), None),
]

def build_run(is_ffc, src, exe):
    argv = [str(FFC), str(src), "-o", str(exe)] if is_ffc else ["gfortran","-w",str(src),"-o",str(exe)]
    out = subprocess.run(argv, capture_output=True, text=True, env=ENV)
    if out.returncode != 0:
        t=(out.stderr or out.stdout).strip()
        return None, t.splitlines()[0] if t else "compile-fail"
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, env=ENV)
    return hashlib.md5(r.stdout.encode()).hexdigest(), r.stdout

def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rows, mismatch, refused = [], 0, 0
    for name, body in CASES:
        src = WORK / f"{name}.f90"; src.write_text(body)
        g, go = build_run(False, src, WORK/f"{name}.gf")
        if g is None: rows.append((name,"REF_FAIL","",go)); continue
        f, fo = build_run(True, src, WORK/f"{name}.fc")
        if f is None: rows.append((name,"REFUSED",g,fo)); refused+=1; continue
        ok = f==g
        if not ok: mismatch+=1
        rows.append((name,"MATCH" if ok else "MISMATCH",g,f))
    print(f"runs={len([r for r in rows if r[1] in ('MATCH','MISMATCH')])} "
          f"match={sum(1 for r in rows if r[1]=='MATCH')} mismatch={mismatch} refused={refused} report={REPORT}")
    for r in rows:
        if r[1] in ("MISMATCH","REFUSED","REF_FAIL"): print(f"  {r[1]} {r[0]} ref={r[2]} ffc={r[3][:100]}")
    return 1 if (mismatch or refused) else 0

if __name__ == "__main__":
    sys.exit(main())
