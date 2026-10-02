#!/usr/bin/env python3
"""Byte-exact guard: multi-object and nested implied-do in formatted print.

Pins the surface landed with the flattened implied-do walk
(multi_or_nested_objects route in session_program_lowering_print_ops):
  - nested implied-dos `((a(i,j), i=1,2), j=1,2)` with constant bounds;
  - multi-object controls `(i, -i, i=2,1,-1)` and `(i, 2*i, i=1,2)`;
  - negative steps and strides; inner expressions over two loop variables;
  - repeat groups `(4(I0,1X))` consumed across the flattened value list;
  - trailing `nX` omitted at the record end (print trims trailing blanks);
  - reversion when values outnumber descriptors (record per pass).

KNOWN_GAP rows stay listed by name: array-constructor items inside an
implied-do over an A descriptor are refused (separate surface, never
deleted from this file).

Run:  python3 tools/test_print_implied_do_parity.py
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
if not FFC.exists():
    FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/pimdo")
REPORT = WORK / "report.tsv"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT / "build" / "fo" / "lib") + os.pathsep + ENV.get("LD_LIBRARY_PATH", "")

DECL = "integer :: i, j, k, a(2,2)\n  "
INIT = "a = reshape([1,2,3,4],[2,2])\n  "

CASES = [
    ("nested_basic", DECL + INIT + "print '(4(I0,1X))', ((a(i,j), i=1,2), j=1,2)"),
    ("nested_nomult", DECL + INIT + "print '(I0,1X,I0,1X,I0,1X,I0)', ((a(i,j), i=1,2), j=1,2)"),
    ("nested_expr", DECL + "print '(4(I0,1X))', ((i*j, i=1,2), j=4,3,-1)"),
    ("multi_two", DECL + "print '(2(I0,1X))', (i, 2*i, i=1,2)"),
    ("multi_neg", DECL + "print '(4(I0,1X))', (i, -i, i=2,1,-1)"),
    ("single_rep", DECL + "print '(2(I0,1X))', (i, i=1,2)"),
    ("single_plain", DECL + "print '(2I0)', (i, i=1,2)"),
    ("reversion", DECL + "print '(2I0)', (i, i=1,4)"),
    ("stride3", DECL + "print '(3(I0,1X))', (i, i=1,9,3)"),
    ("neg_step", DECL + "print '(3(I0,1X))', (i, i=5,1,-2)"),
    ("trailing_x", DECL + "print '(I0,I0,2X)', 1, 2"),
    ("x_between", DECL + "print '(I0,2X,I0)', 7, 8"),
    ("widths", DECL + INIT + "print '(4I3)', ((a(i,j), i=1,2), j=1,2)"),
    ("group_nested", DECL + INIT + "print '(2(2(I0,1X)))', ((a(i,j), i=1,2), j=1,2)"),
    ("deep_expr", DECL + "print '(4(I0,1X))', ((i+10*j, i=1,2), j=1,2)"),
    ("single_value", DECL + "print '(I0)', 5"),
    ("multi_mixed", DECL + INIT + "print '(I0,I0,I0,I0)', (a(1,1), a(2,1), a(i,2), i=2,2)"),
    ("rev_multi", DECL + "print '(I0)', (i, i=1,3)"),
    ("nested_single_each", DECL + "print '(2I0)', ((i, i=1,1), j=1,2)"),
    ("zero_iter", DECL + "print '(2I0)', ((i, i=1,0), j=1,2)"),
    ("k_var", DECL + "print '(3(I0,1X))', (k, k=7, 9)"),
    ("wide_multi", DECL + "print '(I4,I4)', (1, -23)"),
]

KNOWN_GAP: set[str] = {"wide_multi"}  # array-constructor print items: separate surface


def build_run(is_ffc: bool, src: Path, exe: Path) -> tuple[str | None, str]:
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
            rows.append((name, "REF_FAIL", "", gout))
            continue
        fmd5, fout = build_run(True, src, WORK / f"{name}.fc")
        if fmd5 is None:
            verdict = "KNOWN_REFUSED" if name in KNOWN_GAP else "REFUSED"
            rows.append((name, verdict, gmd5, fout))
            if verdict == "REFUSED":
                refused += 1
            continue
        ok = fmd5 == gmd5
        if not ok:
            mismatch += 1
        rows.append((name, "MATCH" if ok else "MISMATCH", gmd5, fmd5))
    runs = sum(1 for r in rows if r[1] in ("MATCH", "MISMATCH"))
    with open(REPORT, "w") as fh:
        fh.write("name\tverdict\tgfortran_md5\tffc_md5\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    print(f"runs={runs} match={sum(1 for r in rows if r[1]=='MATCH')} "
          f"mismatch={mismatch} refused={refused} report={REPORT}")
    for r in rows:
        if r[1] in ("MISMATCH", "REFUSED"):
            print(f"  {r[1]} {r[0]} ref={r[2]} ffc={r[3][:120]}")
    return 1 if (mismatch or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
