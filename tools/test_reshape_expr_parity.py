#!/usr/bin/env python3
"""Byte-exact guard: reshape(constant, constant-shape) in expression masks.

Pins the mask-side fold (classify_mask_side in
session_program_lowering_logical_reduction): reshape fills column-major,
which is linear storage order, so the reshaped constant array presents the
literal's flat sequence with the new shape. any/all comparisons against
`a /= reshape([..],[..])` lower through the literal mask side (mode 3).

Rows cover: ==, /=, > with reshape rhs; 2-D and rank-3 shapes; reshape on
the LEFT side; all()/count(); nested expressions; truthy/false outcomes;
non-square shapes; the fortfront example pattern verbatim.
Refusal rows stay named: non-constant shape, order=/pad= kwargs.

Run:  python3 tools/test_reshape_expr_parity.py
Exit: 0 all rows match or are named gaps; 1 otherwise.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "bin" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/rsx")
REPORT = WORK / "report.tsv"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT / "build" / "fo" / "lib") + os.pathsep + ENV.get("LD_LIBRARY_PATH", "")

D = "integer :: a(2,2), b(3), c(6)\n  a = reshape([1,2,3,4],[2,2])\n  b = reshape([7,8,9],[3])\n  c = reshape([1,2,3,4,5,6],[6])\n  "

CASES = [
    ("ne_true", D + "if (any(a /= reshape([2,-2,1,-1],[2,2]))) print *, 'DIFF'"),
    ("ne_false", D + "if (any(a /= reshape([1,2,3,4],[2,2]))) print *, 'DIFF'\n  print *, 'OK'"),
    ("eq_true", D + "if (all(a == reshape([1,3,2,4],[2,2]))) print *, 'SAME'"),
    ("eq_false", D + "if (all(a == reshape([9,9,9,9],[2,2]))) print *, 'SAME'\n  print *, 'NO'"),
    ("gt_any", D + "if (any(a > reshape([0,0,0,0],[2,2]))) print *, 'GT'"),
    ("lt_none", D + "if (any(a < reshape([0,0,0,0],[2,2]))) print *, 'LT'\n  print *, 'NONE'"),
    ("rank1_b", D + "if (any(b /= reshape([7,8,10],[3]))) print *, 'DIFF'"),
    ("count_eq", D + "print '(I0)', count(a == reshape([1,2,3,4],[2,2]))"),
    ("count_gt", D + "print '(I0)', count(a > reshape([0,2,0,2],[2,2]))"),
    ("rank3", "integer :: r(2,1,3)\n  r = reshape([1,2,3,4,5,6],[2,1,3])\n  if (any(r == reshape([1,2,3,4,5,6],[2,1,3]))) print *, 'OK'"),
    ("rect_shape", "integer :: m(3,2)\n  m = reshape([1,2,3,4,5,6],[3,2])\n  print '(6I2)', count(m == reshape([1,2,3,4,5,6],[3,2]))"),
    ("c_rank1", D + "if (any(c /= reshape([1,2,3,4,5,7],[6]))) print *, 'LAST'"),
    ("all_gt", D + "if (all(a > reshape([0,0,0,0],[2,2]))) print *, 'ALLGT'"),
    ("example_pat", D + "if (any(a /= reshape([2,-2,1,-1],[2,2]))) print *, 'STOPPED'"),
    ("reshape_eq_lit", D + "if (all(reshape([1,2,3,4],[2,2]) == a)) print *, 'L'"),
    ("neg_vals", D + "if (any(a /= reshape([-1,-2,-3,-4],[2,2]))) print *, 'NE'"),
    ("count_ne", D + "print '(I0)', count(b /= reshape([7,0,9],[3]))"),
    ("nested_add_cmp", D + "print '(I0)', count(a + 1 > reshape([2,3,4,5],[2,2]))"),
    ("reshape_ge", D + "if (any(a >= reshape([1,2,3,5],[2,2]))) print *, 'GE'"),
    ("count_all_ne", D + "print '(I0)', count(c /= reshape([6,5,4,3,2,1],[6]))"),
]

REFUSE_CASES = [
    ("refuse_dyn_shape", "integer :: n(2)\n  " + D + "  n = [2,2]\n  print '(I0)', count(a == reshape([1,2,3,4], n))\n", 'reshape'),
    ("refuse_sum_expr", D + "print '(I0)', count(a + reshape([1,1,1,1],[2,2]) > reshape([2,3,4,5],[2,2]))\n", 'not an identifier'),
    ("refuse_or_mask", D + "if (any(a == reshape([1,9,9,9],[2,2]) .or. a == reshape([9,9,9,4],[2,2]))) print *, 'EDGE'\n", 'not an identifier'),
    ("refuse_and_mask", D + "if (any(a > reshape([0,1,0,1],[2,2]) .and. a < reshape([9,9,9,9],[2,2]))) print *, 'MID'\n", 'not an identifier'),
    ("refuse_order", D + "print '(I0)', count(a == reshape([1,2,3,4],[2,2],order=[2,1]))\n", 'reshape'),
]


def build_run(is_ffc: bool, src: Path, exe: Path):
    argv = [str(FFC), str(src), "-o", str(exe)] if is_ffc else \
        ["gfortran", "-w", str(src), "-o", str(exe)]
    out = subprocess.run(argv, capture_output=True, text=True, env=ENV)
    if out.returncode != 0:
        text = (out.stderr or out.stdout).strip()
        return None, text.splitlines()[0] if text else "compile-fail"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, env=ENV)
    return hashlib.md5(run.stdout.encode()).hexdigest(), run.stdout


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rows, mismatch, refused = [], 0, 0
    for name, body in CASES:
        src = WORK / f"{name}.f90"
        src.write_text(f"program p\n  implicit none\n  {body}\nend program p\n")
        gmd5, gout = build_run(False, src, WORK / f"{name}.gf")
        if gmd5 is None:
            rows.append((name, "REF_FAIL", "", gout)); continue
        fmd5, fout = build_run(True, src, WORK / f"{name}.fc")
        if fmd5 is None:
            rows.append((name, "REFUSED", gmd5, fout)); refused += 1; continue
        ok = fmd5 == gmd5
        if not ok: mismatch += 1
        rows.append((name, "MATCH" if ok else "MISMATCH", gmd5, fmd5))
    for name, body, frag in REFUSE_CASES:
        src = WORK / f"{name}.f90"
        src.write_text(f"program p\n  implicit none\n  {body}end program p\n")
        fmd5, fout = build_run(True, src, WORK / f"{name}.fc")
        if fmd5 is None and frag in fout.lower():
            rows.append((name, "REFUSED_AS_PINNED", "", fout))
        else:
            rows.append((name, "BAD_ACCEPT" if fmd5 else "WRONG_MSG", "", fout))
            refused += 1
    runs = sum(1 for r in rows if r[1] in ("MATCH", "MISMATCH"))
    with open(REPORT, "w") as fh:
        fh.write("name\tverdict\tgfortran_md5\tffc_md5\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"runs={runs} match={sum(1 for r in rows if r[1]=='MATCH')} "
          f"mismatch={mismatch} refused={refused} report={REPORT}")
    for r in rows:
        if r[1] in ("MISMATCH", "REFUSED", "BAD_ACCEPT", "WRONG_MSG"):
            print(f"  {r[1]} {r[0]} ref={r[2]} ffc={r[3][:120]}")
    return 1 if (mismatch or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
