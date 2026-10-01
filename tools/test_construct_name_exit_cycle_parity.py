#!/usr/bin/env python3
"""Guard the EXIT/CYCLE construct-name surface (ffc#455).

Named EXIT/CYCLE that name an ENCLOSING loop are silently mis-bound to the
innermost loop: `outer: do; do; if (j==1) exit outer` exits the INNER loop, the
outer loop keeps iterating, and the program prints rows gfortran never prints.
The label is captured by FortFront on the statement (`exit_node%label`,
`cycle_node%label`) but the DO node carries no construct name, and ffc keeps only
a scalar innermost exit/latch block - so there is nothing to resolve a name
against. See ffc#455 for the full chain and the forced fix order.

This guard therefore pins the half that is CORRECT today - unnamed EXIT/CYCLE at
every nesting depth, and names on the innermost loop - because a fix for #455
touches exactly this code and must not regress it. The enclosing-named rows are
reported as `KNOWN_GAP`: they compile and run, and print the wrong thing. They
are kept visible without turning the guard red, so the count of known wrong
answers is a number that can only go down.

Rows whose reference build refuses are `REF_FAIL` (reported, skipped, never a
pass). A byte mismatch against gfortran outside KNOWN_GAP fails the run.

Run:  python3 tools/test_construct_name_exit_cycle_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/cnname")
REPORT = Path("/var/tmp/ffc-goal/perf/cnname/report.tsv")

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

# Wrong today (mis-bound to innermost), kept visible, not failed.
KNOWN_GAP = [
    # Mis-bound to the innermost loop: #455.
    "outer_exit", "outer_cycle", "outer_exit_named_deep", "outer_cycle_named_deep",
    # Refused outright: named DO nested >=2 levels below unnamed DOs, #760.
    "innermost_named_deep", "innermost_named_exit_deep",
]

GAP = [
    ("innermost_named_exit_deep", "program p\n  do i=1,2\n    do j=1,3\n      e: do k=1,3\n        if (k==2) exit e\n        print *, i*100+j*10+k\n      end do e\n    end do\n  end do\nend program p\n"),
    ("innermost_named_deep", "program p\n  do i=1,2\n    do j=1,2\n      c: do k=1,3\n        if (k==2) cycle c\n        print *, i, j, k\n      end do c\n    end do\n  end do\nend program p\n"),
    ("outer_exit", "program p\n  outer: do i=1,3\n    do j=1,3\n      if (j==1) exit outer\n      print *, 100+i, 10*j\n    end do\n    print *, 7\n  end do outer\nend program p\n"),
    ("outer_cycle", "program p\n  outer: do i=1,3\n    do j=1,3\n      if (j==1) cycle outer\n      print *, 100+i, 10*j\n    end do\n  end do outer\nend program p\n"),
    ("outer_exit_named_deep", "program p\n  a: do i=1,2\n    b: do j=1,2\n      do k=1,3\n        if (k==1) exit a\n        print *, i*100+j*10+k\n      end do\n      print *, 7\n    end do b\n  end do a\nend program p\n"),
    ("outer_cycle_named_deep", "program p\n  a: do i=1,2\n    b: do j=1,2\n      do k=1,3\n        if (k==1) cycle a\n        print *, i*100+j*10+k\n      end do\n    end do b\n  end do a\nend program p\n"),
]


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
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def check(name: str, src_text: str, lines: list[str]) -> tuple[int, int, int]:
    src = WORK / f"{name}.f90"
    src.write_text(free_form(src_text))
    rc_g, _, _ = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
    if rc_g != 0:
        lines.append(f"{name}\tREF_FAIL")
        return 0, 0, 0
    rc_f, _, err = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
    if rc_f != 0:
        if name in KNOWN_GAP:
            lines.append(f"{name}\tKNOWN_GAP\tREFUSED_NOW")
            return 0, 0, 0
        # A refusal on a currently-working form is a regression.
        lines.append(f"{name}\tREGRESSED_REFUSED\t{err.strip()[:48]}")
        return 1, 0, 1
    _, rout, _ = run([str(WORK / f"{name}_ref")])
    ffrc, fout, _ = run([str(WORK / f"{name}_ffc")])
    rmd5 = hashlib.md5(rout.encode()).hexdigest()
    fmd5 = hashlib.md5(fout.encode()).hexdigest()
    if ffrc != 0:
        lines.append(f"{name}\tFFC_RUNTIME_FAIL")
        return 1, 0, 1
    if rmd5 == fmd5:
        lines.append(f"{name}\tMATCH\t{rmd5}")
        return 1, 1, 0
    if name in KNOWN_GAP:
        lines.append(f"{name}\tKNOWN_GAP\tMISMATCH_wrong")
        return 0, 0, 0
    lines.append(f"{name}\tMISMATCH\tref={rmd5[:12]}\tffc={fmd5[:12]}")
    return 1, 0, 1


def main() -> int:
    if not FFC.exists():
        print(f"SKIP: ffc not built at {FFC}", file=sys.stderr)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    runs = match = fail = 0
    for name, text in GUARDED + GAP:
        r, m, f = check(name, text, lines)
        runs += r
        match += m
        fail += f
    gap_rows = sum(1 for l in lines if l.split("\t")[1] == "KNOWN_GAP")
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} fail={fail} known_gap={gap_rows}")
    print(f"report={REPORT}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
