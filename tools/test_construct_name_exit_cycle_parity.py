#!/usr/bin/env python3
"""Compare structured DO branch targets and scalar snapshots with gfortran.

Named EXIT/CYCLE resolve enclosing counted, while, and infinite loops (ffc#455).
Two named loops nested beneath unnamed loops remain tracked as known parser
refusals (ffc#760). Every accepted case compares complete output bytes and
records both MD5 digests; all other mismatches and refusals fail the oracle.

Run: python3 tools/test_construct_name_exit_cycle_parity.py
Falsify comparisons: python3 tools/test_construct_name_exit_cycle_parity.py --falsify
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path(os.environ.get("FFC_PARITY_WORK", "/var/tmp/ffc-goal/perf/cnname"))
REPORT = WORK / "report.tsv"
FALSIFY = False

# Correct today - these MUST keep matching.
GUARDED = [
    ("plain_exit", "program p\n  do i=1,3\n    if (i==2) exit\n    print *, i\n  end do\nend program p\n"),
    ("plain_cycle", "program p\n  do i=1,4\n    if (mod(i,2)==0) cycle\n    print *, i\n  end do\nend program p\n"),
    ("inner_exit", "program p\n  i0: do i=1,3\n    print *, i\n    if (i==2) exit i0\n  end do i0\nend program p\n"),
    ("inner_cycle", "program p\n  i0: do i=1,4\n    if (i==2) cycle i0\n    print *, i\n  end do i0\nend program p\n"),
    ("inner_cycle_lbl", "program p\n  inner: do j=1,3\n    if (j==2) cycle inner\n    print *, j\n  end do inner\nend program p\n"),
    ("inner_exit_lbl", "program p\n  inner: do j=1,3\n    if (j==2) exit inner\n    print *, j\n  end do inner\nend program p\n"),
    ("nest_plain_exit", "program p\n  do i=1,3\n    do j=1,3\n      if (j==2) exit\n      print *, i, j\n    end do\n    print *, 7\n  end do\nend program p\n"),
    ("nest_plain_cycle", "program p\n  do i=1,3\n    do j=1,3\n      if (j==2) cycle\n      print *, i, j\n    end do\n  end do\nend program p\n"),
    ("deep_plain_exit", "program p\n  do i=1,2\n    do j=1,2\n      do k=1,3\n        if (k==2) exit\n        print *, i, j, k\n      end do\n    end do\n  end do\nend program p\n"),
    ("deep_plain_cycle", "program p\n  do i=1,3\n    do j=1,2\n      do k=1,3\n        if (k==2) cycle\n        print *, i*100+j*10+k\n      end do\n    end do\n  end do\nend program p\n"),


    ("dowhile_plain_exit", "program p\n  integer :: i\n  i=0\n  do while (i<4)\n    i=i+1\n    if (i==3) exit\n    print *, i\n  end do\nend program p\n"),
    ("dowhile_named_exit", "program p\n  integer :: i\n  i=0\n  w: do while (i<4)\n    i=i+1\n    if (i==3) exit w\n    print *, i\n  end do w\nend program p\n"),
    ("dowhile_named_cycle", "program p\n  integer :: i\n  i=0\n  w: do while (i<5)\n    i=i+1\n    if (i==2) cycle w\n    print *, i\n  end do w\nend program p\n"),
    ("exit_after_inner", "program p\n  do i=1,2\n    do j=1,2\n      if (j==2) exit\n      print *, i*10+j\n    end do\n    print *, 99\n  end do\nend program p\n"),
    ("cycle_accumulate", "program p\n  integer :: s\n  s=0\n  do i=1,5\n    if (i==3) cycle\n    s=s+i\n  end do\n  print *, s\nend program p\n"),
    ("named_cycle_inner_val", "program p\n  integer :: s\n  s=0\n  o: do i=1,4\n    if (i==2) cycle o\n    s=s+i\n  end do o\n  print *, s\nend program p\n"),
    ("named_exit_inner_val", "program p\n  integer :: s\n  s=0\n  o: do i=1,4\n    s=s+i\n    if (i==2) exit o\n  end do o\n  print *, s\nend program p\n"),
    ("nest_inner_named_val", "program p\n  integer :: s\n  s=0\n  do i=1,3\n    o: do j=1,3\n      if (j==2) exit o\n      s=s+1\n    end do o\n  end do\n  print *, s\nend program p\n"),
    ("plain_cycle_val", "program p\n  integer :: s\n  s=0\n  do i=1,6\n    if (i>4) cycle\n    s=s+i\n  end do\n  print *, s\nend program p\n"),
    ("plain_exit_val", "program p\n  integer :: s\n  s=0\n  do i=1,6\n    s=s+i\n    if (i>3) exit\n  end do\n  print *, s\nend program p\n"),
    ("nest_plain_inner_lbl", "program p\n  do i=1,2\n    z: do j=1,3\n      if (j==3) exit z\n      print *, i*10+j\n    end do z\n  end do\nend program p\n"),
    ("dowhile_plain_cycle", "program p\n  integer :: i\n  i=0\n  do while (i<5)\n    i=i+1\n    if (i==2) cycle\n    print *, i\n  end do\nend program p\n"),
]

# Parser refusals are a separate frontend issue, never output exemptions.
KNOWN_REFUSED = {"innermost_named_deep", "innermost_named_exit_deep"}

GAP = [
    ("innermost_named_exit_deep", "program p\n  do i=1,2\n    do j=1,3\n      e: do k=1,3\n        if (k==2) exit e\n        print *, i*100+j*10+k\n      end do e\n    end do\n  end do\nend program p\n"),
    ("innermost_named_deep", "program p\n  do i=1,2\n    do j=1,2\n      c: do k=1,3\n        if (k==2) cycle c\n        print *, i, j, k\n      end do c\n    end do\n  end do\nend program p\n"),
    ("outer_exit", "program p\n  outer: do i=1,3\n    do j=1,3\n      if (j==1) exit outer\n      print *, 100+i, 10*j\n    end do\n    print *, 7\n  end do outer\nend program p\n"),
    ("outer_cycle", "program p\n  outer: do i=1,3\n    do j=1,3\n      if (j==1) cycle outer\n      print *, 100+i, 10*j\n    end do\n  end do outer\nend program p\n"),
    ("outer_exit_named_deep", "program p\n  a: do i=1,2\n    b: do j=1,2\n      do k=1,3\n        if (k==1) exit a\n        print *, i*100+j*10+k\n      end do\n      print *, 7\n    end do b\n  end do a\nend program p\n"),
    ("outer_cycle_named_deep", "program p\n  a: do i=1,2\n    b: do j=1,2\n      do k=1,3\n        if (k==1) cycle a\n        print *, i*100+j*10+k\n      end do\n    end do b\n  end do a\nend program p\n"),
]


TARGET_VALUES = []
for loop_kind in ("counted", "while", "infinite"):
    for branch in ("exit", "cycle"):
        opening = {"counted": "outer: do i=1,4",
                   "while": "outer: do while (i<4)",
                   "infinite": "outer: do"}[loop_kind]
        advance = "" if loop_kind == "counted" else "    i=i+1\n"
        terminate = "    if (i>4) exit outer\n" if loop_kind == "infinite" else ""
        source = ("program p\n  integer :: i,j,s\n  i=0\n  s=0\n  " + opening + "\n"
                  + advance + terminate + "    s=s+i\n    do j=1,3\n"
                  "      s=s+10\n      if (j==2) " + branch + " outer\n"
                  "      s=s+1\n    end do\n    s=s+1000\n  end do outer\n"
                  "  print *, i,j,s\nend program p\n")
        TARGET_VALUES.append((f"outer_{loop_kind}_{branch}_values", source))

for branch in ("exit", "cycle"):
    TARGET_VALUES.append((f"outer_counted_inner_while_{branch}",
        "program p\n  integer :: i,j,s\n  s=0\n  outer: do i=1,4\n"
        "    j=0\n    do while (j<3)\n      j=j+1\n      s=s+10\n"
        f"      if (j==2) {branch} outer\n"
        "      s=s+1\n    end do\n    s=s+1000\n  end do outer\n"
        "  print *, i,j,s\nend program p\n"))
    TARGET_VALUES.append((f"middle_{branch}_values",
        "program p\n  integer :: i,j,k,s\n  s=0\n  a: do i=1,2\n"
        "    b: do j=1,3\n      do k=1,3\n        s=s+100*i+10*j+k\n"
        f"        if (k==2) {branch} b\n"
        "      end do\n      s=s+1000\n    end do b\n    s=s+10000\n"
        "  end do a\n  print *, i,j,k,s\nend program p\n"))
    TARGET_VALUES.append((f"outer_{branch}_integer8",
        "program p\n  integer :: i,j\n  integer(8) :: s\n"
        "  s=5000000000_8\n  outer: do i=1,4\n    do j=1,3\n"
        "      s=s+1000000000_8\n"
        f"      if (j==2) {branch} outer\n"
        "    end do\n    s=s+1_8\n  end do outer\n"
        "  print *, i,j,s\nend program p\n"))

TARGET_VALUES.append(("outer_mixed_case_cycle",
    "program p\n  integer :: i,j,s\n  s=0\n  Outer: do i=1,3\n"
    "    do j=1,3\n      s=s+1\n      if (j==2) cycle OUTER\n"
    "    end do\n  end do Outer\n  print *, i,j,s\nend program p\n"))


def free_form(src: str) -> str:
    """Split a packed one-line program into free-form lines.

    Free-form Fortran caps a line at 132 characters. Several cases below were
    written as one long line, so the compiler saw truncated text and reported
    "Unrecognized statement: end program p" - a bug in this generator, not in
    ffc. Breaking the line fixes the oracle without touching what it tests.
    """
    if all(len(l) <= 132 for l in src.split("\n")):
        return src
    out = src
    for tok in ("  do ", "    do ", "    if (", "      if (", "        if (",
                "      print ", "    print ", "  print ", "  end do",
                "    end do", "      end do", "        end do", "end program"):
        out = out.replace(tok, "\n" + tok.lstrip())
    out = out.replace("\n\n", "\n")
    head, *rest = out.split("\n")
    body = [l for l in rest if l.strip()]
    return head + "\n" + "\n".join("  " + l.strip() for l in body) + "\n"


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def check(name: str, src_text: str, lines: list[str]) -> tuple[int, int, int]:
    src = WORK / f"{name}.f90"
    src.write_text(free_form(src_text))
    rc_g, _, _ = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
    if rc_g != 0:
        lines.append(f"{name}\tREF_FAIL")
        return 0, 0, 0
    rc_f, _, err = run(["fo", "exec", "--no-build", "ffc", str(src),
                        "-o", str(WORK / f"{name}_ffc")])
    if rc_f != 0:
        if name in KNOWN_REFUSED:
            lines.append(f"{name}\tKNOWN_REFUSED\tffc#760")
            return 0, 0, 0
        # A refusal on a currently-working form is a regression.
        lines.append(f"{name}\tREGRESSED_REFUSED\t{err.strip()[:48]}")
        return 1, 0, 1
    refrc, rout, _ = run([str(WORK / f"{name}_ref")])
    if refrc != 0:
        lines.append(f"{name}\tREF_RUNTIME_FAIL")
        return 1, 0, 1
    ffrc, fout, _ = run([str(WORK / f"{name}_ffc")])
    if FALSIFY:
        fout += "falsified candidate output\n"
    rmd5 = hashlib.md5(rout.encode()).hexdigest()
    fmd5 = hashlib.md5(fout.encode()).hexdigest()
    if ffrc != 0:
        lines.append(f"{name}\tFFC_RUNTIME_FAIL")
        return 1, 0, 1
    if rmd5 == fmd5:
        lines.append(f"{name}\tMATCH\tref={rmd5}\tffc={fmd5}")
        return 1, 1, 0
    lines.append(f"{name}\tMISMATCH\tref={rmd5}\tffc={fmd5}")
    return 1, 0, 1


def main() -> int:
    global FALSIFY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--falsify", action="store_true",
                        help="inject wrong candidate bytes to prove the oracle fails")
    FALSIFY = parser.parse_args().falsify
    if not FFC.exists():
        print(f"FAIL: ffc not built at {FFC}; run fo build", file=sys.stderr)
        return 1
    WORK.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    runs = match = fail = 0
    for name, text in GUARDED + GAP + TARGET_VALUES:
        r, m, f = check(name, text, lines)
        runs += r
        match += m
        fail += f
    gap_rows = sum(1 for l in lines if l.split("\t")[1] == "KNOWN_REFUSED")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} fail={fail} known_refused={gap_rows}")
    print(f"report={REPORT}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
