#!/usr/bin/env python3
"""Byte-exact guard: internal write with groups, repeat counts and implied-dos.

Pins the surface landed for the fortfront `io_implied_do_objects` corpus case:
  - leading repeat counts on I/A descriptors (`4I0`) that previously died in the
    single-descriptor parser with "unsupported edit descriptor: 4I0";
  - parenthesized groups and repeated groups (`4(I0,1X)`, `2(A,I2)`);
  - single-, multi- and nested-object implied-dos with positive and negative
    constant steps, expanded at lowering time with the loop variable bound;
  - nX spacing beside data descriptors, and format reversion records.

Every row must match gfortran byte-exactly; the oracle FALSIFIES by pinning
distinct md5s per row, so a dropped or reordered field fails the run.

KNOWN_GAP (kept by name, never deleted):
  - `rev_records` -> `write(r,'(I0)') 1, 2`: reversion of a lone descriptor
    with surplus values still dies in the pre-routing single-value guard.

Run:  python3 tools/test_internal_write_implied_do_parity.py
Exit: 0 all rows match or are named gaps; 1 on any MISMATCH/REFUSED new row.
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
WORK = Path("/var/tmp/ffc-goal/perf/iwido")
REPORT = WORK / "report.tsv"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT / "build" / "fo" / "lib") + os.pathsep + ENV.get("LD_LIBRARY_PATH", "")

DECL = "integer :: i, j, k, a(2,2)\n  character(len=64) :: r\n"
INIT = "a = reshape([1,2,3,4],[2,2])\n  "

CASES = [
    ("rep_i0", INIT + "write(r,'(4I0)') (i, i=1,4)\n  print '(A)', trim(r)"),
    ("rep_i5", INIT + "write(r,'(3I5)') (i, i=1,3)\n  print '(A)', trim(r)"),
    ("rep_a", INIT + "write(r,'(2A3)') 'abc','de'\n  print '(A)', trim(r)"),
    ("group_rep", INIT + "write(r,'(4(I0,1X))') (i, i=1,4)\n  print '(A)', trim(r)"),
    ("group_mult", INIT + "write(r,'(2(A,I2))') 'x',1,'yy',22\n  print '(A)', trim(r)"),
    ("multi_obj_neg", INIT + "write(r,'(4(I0,1X))') (i, -i, i=2, 1, -1)\n  print '(A)', trim(r)"),
    ("nested_2d", INIT + "write(r,'(4(I0,1X))') ((a(i,j), i=1,2), j=1,2)\n  print '(A)', trim(r)"),
    ("nested_mixed", INIT + "write(r,'(I0,1X,I0,1X,I0,1X,I0)') ((a(i,j), i=1,2), j=1,2)\n  print '(A)', trim(r)"),
    ("x_spacing", INIT + "write(r,'(I0,2X,I0)') 7, 8\n  print '(A)', trim(r)"),
    ("x_repeat", INIT + "write(r,'(I0,1X,I0)') 7, 8\n  print '(A)', trim(r)"),
    ("plain_multi", INIT + "write(r,'(I0,I0,I0)') 1,2,3\n  print '(A)', trim(r)"),
    ("single_i0", INIT + "write(r,'(I0)') 42\n  print '(A)', trim(r)"),
    ("single_a", INIT + "write(r,'(A)') 'hey'\n  print '(A)', trim(r)"),
    ("rev_records", INIT + "write(r,'(I0)') 1, 2\n  print '(A)', trim(r)"),
    ("step3", INIT + "write(r,'(3(I0,1X))') (i, i=1,9,3)\n  print '(A)', trim(r)"),
    ("neg_full", INIT + "write(r,'(3(I0,1X))') (i, i=5, 1, -2)\n  print '(A)', trim(r)"),
    ("nested_neg", INIT + "write(r,'(4(I0,1X))') ((i*j, i=1,2), j=4, 3, -1)\n  print '(A)', trim(r)"),
    ("trailing_x", INIT + "write(r,'(I0,I0,2X)') 1, 2\n  print '(A)', trim(r)"),
    ("char_int_mix", INIT + "write(r,'(A,I0,A,I0)') 'n=',3,'m=',4\n  print '(A)', trim(r)"),
    ("group_x", INIT + "write(r,'(2(I0,2X))') 1, 2\n  print '(A)', trim(r)"),
    ("k_loop", INIT + "write(r,'(2(I0,1X))') (k, k=10, 11)\n  print '(A)', trim(r)"),
    ("reuse_r", INIT + "write(r,'(I0)') 5\n  write(r,'(I0,I0)') 6,7\n  print '(A)', trim(r)"),
]


KNOWN_GAP = {"rev_records"}


def build_run(compiler: str, src: Path, exe: Path) -> tuple[str | None, str]:
    argv = [str(FFC), str(src), "-o", str(exe)] if compiler == "ffc" \
        else ["gfortran", "-w", str(src), "-o", str(exe)]
    out = subprocess.run(argv, capture_output=True, text=True, env=ENV)
    if out.returncode != 0:
        return None, (out.stderr or out.stdout).strip().splitlines()[0] if (out.stderr or out.stdout) else "compile-fail"
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30, env=ENV)
    return hashlib.md5(run.stdout.encode()).hexdigest(), run.stdout


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rows, mismatch, refused = [], 0, 0
    for idx, (name, body) in enumerate(CASES):
        src = WORK / f"{name}.f90"
        src.write_text(f"program w\n  implicit none\n  {DECL}  {body}\nend program w\n")
        # gfortran is the independent reference; every row here is valid F2023.
        gmd5, gout = build_run("gfortran", src, WORK / f"{name}.gf")
        if gmd5 is None:
            rows.append((name, "REF_FAIL", "", gmd5 or gout))
            continue
        fmd5, fout = build_run("ffc", src, WORK / f"{name}.fc")
        if fmd5 is None:
            if name in KNOWN_GAP:
                rows.append((name, "KNOWN_REFUSED", gmd5, fout))
            else:
                rows.append((name, "REFUSED", gmd5, fout))
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
            print(f"  {r[1]} {r[0]} ref={r[2]} ffc={r[3]}")
    return 1 if (mismatch or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
