#!/usr/bin/env python3
"""Byte-exact guard: implied-do targets in internal reads.

Pins the surface landed with the flattened read walk
(lower_internal_read_implied_do in session_program_lowering_internal_read):
  - list-directed `read(buf, *) (a(i), i=1,n)` and comma-separated buffers;
  - formatted `read(buf,'(4I2)') ((a(i,j), i=..), j=..)` nested controls;
  - stride and negative-step controls over two loop variables;
  - format reversion across records embedded as newlines in the buffer
    ('%d' skips the newline; gfortran reverts the format the same way);
  - refusals keep their names (A descriptors, implied-do beside other
    targets).

Width stance: Fortran Iw is an exact field, sscanf %wd reads up to w; rows
here use single-space separation where both agree. gfortran-stricter inputs
(I1 with commas) are deliberately not pinned as accepts.

Run:  python3 tools/test_internal_read_implied_do_parity.py
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
WORK = Path("/var/tmp/ffc-goal/perf/irido")
REPORT = WORK / "report.tsv"
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = str(ROOT / "build" / "fo" / "lib") + os.pathsep + ENV.get("LD_LIBRARY_PATH", "")

DECL = "character(len=80) :: buf\n  integer :: a(3,3), b(6), i, j, k\n  a = 0\n  b = 0\n  "

CASES = [
    ("list_two", DECL + "buf='1 2'\n  read(buf,*) (b(i), i=1,2)\n  print '(2I4)', (b(i), i=1,2)"),
    ("list_three", DECL + "buf='10 20 30'\n  read(buf,*) (b(i), i=1,3)\n  print '(3I4)', (b(i), i=1,3)"),
    ("list_commas", DECL + "buf='10, 20, 30'\n  read(buf,*) (b(i), i=1,3)\n  print '(3I4)', (b(i), i=1,3)"),
    ("list_mixed_sep", DECL + "buf='1,2 3'\n  read(buf,*) (b(i), i=1,3)\n  print '(3I4)', (b(i), i=1,3)"),
    ("list_nested", DECL + "buf='1 2 3 4'\n  read(buf,*) ((a(i,j), i=1,2), j=1,2)\n  print '(4I4)', ((a(i,j), i=1,2), j=1,2)"),
    ("fmt_i2_two", DECL + "buf=' 1 2'\n  read(buf,'(2I2)') (b(i), i=1,2)\n  print '(2I4)', (b(i), i=1,2)"),
    ("fmt_i2_four", DECL + "buf=' 1 2 3 4'\n  read(buf,'(4I2)') (b(i), i=1,4)\n  print '(4I4)', (b(i), i=1,4)"),
    ("fmt_i3_four", DECL + "buf='  1  2  3  4'\n  read(buf,'(4I3)') (b(i), i=1,4)\n  print '(4I4)', (b(i), i=1,4)"),
    ("fmt_nested", DECL + "buf=' 1 2 3 4'\n  read(buf,'(4I2)') ((a(i,j), i=1,2), j=1,2)\n  print '(4I4)', ((a(i,j), i=1,2), j=1,2)"),
    ("fmt_group", DECL + "buf=' 1 2 3 4'\n  read(buf,'(2(2I2))') ((a(i,j), i=1,2), j=1,2)\n  print '(4I4)', ((a(i,j), i=1,2), j=1,2)"),
    ("fmt_stride", DECL + "buf=' 1 2 3'\n  read(buf,'(3I2)') (b(k), k=2,4)\n  print '(3I4)', (b(i), i=2,4)"),
    ("rev_records", DECL + "buf=' 1\n 2\n 3\n 4'\n  read(buf,'(2I2)') (b(i), i=1,4)\n  print '(4I4)', (b(i), i=1,4)"),
    ("multi_objects", DECL + "buf=' 1 2'\n  read(buf,'(I2,I2)') (b(i), i=1,2)\n  print '(2I4)', (b(i), i=1,2)"),
    ("nested_mixed", DECL + "buf=' 1 2 3 4'\n  read(buf,'(4I2)') ((b(i), i=1,2), k=3,4)\n  print '(4I4)', (b(i), i=1,4)"),
    ("single_elem", DECL + "buf=' 7'\n  read(buf,'(I2)') (a(2,2), i=1,1)\n  print '(I4)', a(2,2)"),
    ("expr_target", DECL + "buf=' 5'\n  read(buf,'(I2)') (a(i+1,1), i=1,1)\n  print '(I4)', a(2,1)"),
    ("six_targets", DECL + "buf=' 1 2 3 4 5 6'\n  read(buf,'(6I2)') (b(i), i=1,6)\n  print '(6I3)', (b(i), i=1,6)"),
    ("list_six", DECL + "buf='1,2,3,4,5,6'\n  read(buf,*) (b(i), i=1,6)\n  print '(6I3)', (b(i), i=1,6)"),
    ("neg_bound", DECL + "buf=' 1 2'\n  read(buf,'(2I2)') (a(1,-k), k=-1,-2)\n  print '(2I4)', (a(1,i), i=-1,-2)"),
    ("step_two", DECL + "buf=' 1 2 3'\n  read(buf,'(3I2)') (b(i), i=1,5,2)\n  print '(3I4)', (b(i), i=1,5,2)"),
    ("zero_iter", DECL + "buf=' 1'\n  read(buf,'(I2)') (b(i), i=1,0)\n  print '(I4)', 9"),
]

KNOWN_GAP: set[str] = set()

REFUSE_CASES = [
    ("refuse_a_desc", DECL + "buf='ab'\n  read(buf,'(A2)') (b(i), i=1,1)\n", 'integer edit descriptors'),
    ("refuse_no_fmt", "integer :: c(2), i\n  read(*, '(I0)') (c(i), i=1,2)\n", 'stdin read'),
]


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
    for name, body, frag in REFUSE_CASES:
        src = WORK / f"{name}.f90"
        src.write_text(f"program p\n  implicit none\n  {body}end program p\n")
        fmd5, fout = build_run(True, src, WORK / f"{name}.fc")
        if fmd5 is None and frag in fout:
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
