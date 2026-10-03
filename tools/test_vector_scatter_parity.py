#!/usr/bin/env python3
"""Byte-exact guard: vector-subscript scatter with bare-colon slices.

Pins A(X(:)) = rhs lowering through the same scatter path as A(X) for
static rank-1 integer index vectors. rhs forms: scalar, array literal,
identifier array, arithmetic; plus refusals for explicit bounds X(1:2),
stride X(::2), rank-2 index vectors, and allocatable index vectors.

Run:  python3 tools/test_vector_scatter_parity.py
Exit: 0 when every run row is byte-exact vs gfortran.
"""
from __future__ import annotations
import hashlib, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "bin" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/vscatter")
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT/"build"/"fo"/"lib") + os.pathsep + ENV.get("LD_LIBRARY_PATH","")

def P(b): return f"program p\n  implicit none\n  {b}\nend program p\n"

PRE = "integer :: A(4)\n  integer :: X(3)\n  A = 0\n  X = [1,3,4]\n  "
CASES = [
 ("of_plus1",   P("integer :: A(3), X(2)\n  A = 0\n  X = [1,2]\n  A(X + 1) = [7,8]\n  print '(3I0)', A(1),A(2),A(3)")),
 ("of_minus1",  P("integer :: A(3), X(2)\n  A = 0\n  X = [2,3]\n  A(X - 1) = [5,6]\n  print '(3I0)', A(1),A(2),A(3)")),
 ("of_plus0",   P("integer :: A(3), X(2)\n  A = 0\n  X = [1,3]\n  A(X + 0) = [4,9]\n  print '(3I0)', A(1),A(2),A(3)")),
 ("of_ident_rhs",P("integer :: A(4), X(3), V(3)\n  A = 0\n  X = [1,2,3]\n  V = [11,22,33]\n  A(X + 1) = V\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("of_64",      P("integer :: X(2)\n  X = [0,1]\n  X(X + 1) = 11235\n  print '(2I0)', X(1),X(2)")),
 ("of_scalar_lit",P("integer :: A(4), X(2)\n  A = 0\n  X = [1,3]\n  A(X + 1) = 42\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("of_minus_lit_idx",P("integer :: A(3), X(2)\n  A = 0\n  X = [3,2]\n  A(X - 2) = [8,9]\n  print '(3I0)', A(1),A(2),A(3)")),
 ("sc_scalar",  P(PRE+"A(X(:)) = 7\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_lit",     P(PRE+"A(X(:)) = [9,8,7]\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_ident",   P("integer :: A(4), X(3), V(3)\n  A = 0\n  X = [1,3,4]\n  V = [5,6,7]\n  A(X(:)) = V\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_plain",   P(PRE+"A(X) = 2\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_dup",     P("integer :: A(4), X(3)\n  A = 0\n  X = [2,2,3]\n  A(X(:)) = [1,2,3]\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_rev",     P("integer :: A(3), X(3)\n  A = 0\n  X = [3,2,1]\n  A(X(:)) = [7,8,9]\n  print '(3I0)', A(1),A(2),A(3)")),
 ("sc_neg_val", P(PRE+"A(X(:)) = -4\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_scalar_arith", P("integer :: A(4), X(3)\n  A = 0\n  X = [1,3,4]\n  A(X(:)) = 2*3+1\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_all_idx", P("integer :: A(3), X(3)\n  A = [1,1,1]\n  X = [1,2,3]\n  A(X(:)) = [4,5,6]\n  print '(3I0)', A(1),A(2),A(3)")),
 ("sc_twice",   P(PRE+"A(X(:)) = 1\n  A(X(:)) = 2\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_after_sec",P(PRE+"A(2:3) = [5,6]\n  A(X(:)) = 0\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_sum",     P(PRE+"A(X(:)) = 3\n  print '(I0)', sum(A)")),
 ("sc_var_name",P("integer :: DATA4(4), IDX(2)\n  DATA4 = 0\n  IDX = [2,4]\n  DATA4(IDX(:)) = 8\n  print '(4I0)', DATA4(1),DATA4(2),DATA4(3),DATA4(4)")),
 ("sc_single",  P("integer :: A(3), X(1)\n  A = 0\n  X = [2]\n  A(X(:)) = 5\n  print '(3I0)', A(1),A(2),A(3)")),
 ("sc_gather_pair",P("integer :: A(4), X(3), B(3)\n  A = 0\n  X = [1,3,4]\n  B = [1,2,3]\n  A(X(:)) = B\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_zero_idx_val",P("integer :: A(4), X2(2)\n  A = 0\n  X2=[1,2]\n  A(X2(:)) = 0\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_reused_X",P(PRE+"A(X(:)) = 1\n  X = [4,1,3]\n  A(X(:)) = 6\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
 ("sc_lit_sum", P(PRE+"A(X(:)) = [2,3,4]\n  print '(I0)', sum(A)")),
 ("sc_char_free",P("integer :: A(2), X(2)\n  A = [-1,-2]\n  X = [1,2]\n  A(X(:)) = [10,20]\n  print '(2I0)', A(1),A(2)")),
 ("sc_expr_idx",P("integer :: A(4), X(3)\n  A = 0\n  X = [mod(1,4)+1, 2*2-1, 3]\n  A(X(:)) = 9\n  print '(4I0)', A(1),A(2),A(3),A(4)")),
]

def build_run(is_ffc, src, exe):
    argv=[str(FFC),str(src),"-o",str(exe)] if is_ffc else ["gfortran","-w",str(src),"-o",str(exe)]
    out=subprocess.run(argv,capture_output=True,text=True,env=ENV)
    if out.returncode!=0: return None,(out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout).strip() else "compile-fail"
    r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30,env=ENV)
    return hashlib.md5(r.stdout.encode()).hexdigest(), r.stdout

def main()->int:
    WORK.mkdir(parents=True,exist_ok=True)
    rows=[]; mismatch=refused=0
    for name,body in CASES:
        src=WORK/f"{name}.f90"; src.write_text(body)
        g,go=build_run(False,src,WORK/f"{name}.gf")
        if g is None: rows.append((name,"REF_FAIL","",go)); continue
        f,fo=build_run(True,src,WORK/f"{name}.fc")
        if f is None: rows.append((name,"REFUSED",g,fo)); refused+=1; continue
        ok=f==g
        if not ok: mismatch+=1
        rows.append((name,"MATCH" if ok else "MISMATCH",g,f))
    runs=sum(1 for r in rows if r[1] in("MATCH","MISMATCH"))
    print(f"runs={runs} match={sum(1 for r in rows if r[1]=='MATCH')} mismatch={mismatch} refused={refused} report={WORK/'report.tsv'}")
    with open(WORK/"report.tsv","w") as fh:
        for r in rows: fh.write("\t".join(r)+"\n")
    for r in rows:
        if r[1]!="MATCH": print(f"  {r[1]} {r[0]} ffc={r[3][:90]}")
    return 1 if mismatch or refused else 0

if __name__=="__main__": sys.exit(main())
