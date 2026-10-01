#!/usr/bin/env python3
"""Regression guard for the runtime-length character surface (PLAN W1 #348).

#348 asks to centralize runtime-length character operations. Before touching
that code the *working* surface was probed so it is pinned, not assumed.
Measured finding: every locally reachable form already matches gfortran -
`trim(x)//y`, `repeat(x,n)//y`, `achar(c)//y`, `len(trim(x))`, a character
function result in concat, `character(len=*)` dummies, fixed-length dummies and
character function arguments. So the XFAIL rows attributed to #348 are NOT in
this print/concat surface; they sit in the lfortran and gfortran.dg corpora,
which are absent from this workspace (see lazy-fortran/ffc#759), or need
literal-base substring ranges in FortFront (fortfront#3018).

That distinction matters: it says the remaining work is descriptor passing and
corpus access, not print lowering, and stops the next reader from re-doing this
probe.  Each row is byte-exact against gfortran.  `ref_fail` counts rows the
reference compiler itself rejects - a compile-time refusal there is a valid
non-goal, so it is reported and skipped, never downgraded into a pass.
`KNOWN_GAP` names rows the reference accepts and ffc refuses, so a real gap
stays visible without turning the guard red.

Run:  python3 tools/test_char_runtime_surface_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/chrsurf")
REPORT = Path("/var/tmp/ffc-goal/perf/chrsurf/report.tsv")

CASES = [
    (decl, body)
    for decl, body in [
        ('character(len=10) :: a\n  a="ab"', 'trim(a)//"Z"'),
        ('character(len=10) :: a\n  a="ab"', 'trim(a)//trim(a)'),
        ('character(len=8) :: a\n  a="x"', 'repeat(a,3)//"y"'),
        ('character(len=8) :: a\n  a="ab"', 'repeat(a,2)//trim(a)'),
        ('character(len=4) :: a\n  a="q"', 'achar(65)//trim(a)'),
        ('character(len=4) :: a\n  a="hello"', "len(trim(a))"),
        ('character(len=6) :: a, b\n  a="foo"\n  b="bar"', "trim(a)//trim(b)"),
        ('character(len=6) :: a, b\n  a="foo"\n  b="bar"', "len(trim(a//b))"),
        ('character(len=5) :: a, b\n  a="abc"\n  b="xy"', 'trim(a)//trim(b)//"Z"'),
        ('character(len=3) :: c(2)\n  c=(/ "abc", "def" /)', 'c(1)(2:3)//c(2)(1:2)'),
        ('character(len=3) :: c(2)\n  c=(/ "abc", "def" /)', 'trim(c(2))//c(1)(1:1)'),
        ('character(len=2) :: c(3)\n  c=(/ "ab", "cd", "ef" /)',
         'c(1)(2:2)//c(2)(1:2)//c(3)(1:1)'),
        ('character(len=7) :: a\n  a="AbCdeFG"', 'a(2:4)//a(5:7)'),
        ('character(len=7) :: a\n  a="AbCdeFG"', 'achar(97)//a(3:4)'),
        ('character(len=4) :: a\n  a="wxyz"', 'repeat(a(1:2),2)//trim(a)'),
        ('character(len=6) :: a\n  a="  pad"', 'trim(adjustl(a))//"."'),
        ('character(len=6) :: a\n  a="pad  "', 'trim(adjustr(a))//"!"'),
    ]
]
# Runtime-length character dummies and function results: the descriptor-passing
# surface named in #348. Probed as working, pinned so an ABI change cannot
# silently break it.
DUMMY_CASES = [
    ("dummy_star", 'character(len=*) :: x', 'print *, trim(x)',
     'character(len=5) :: s\n  s="hello"\n  call sub(s)'),
    ("dummy_fixed", 'character(len=3) :: x', 'print *, x',
     'character(len=3) :: s\n  s="abc"\n  call sub(s)'),
    ("dummy_concat", 'character(len=*) :: x, y', 'print *, trim(x)//trim(y)',
     'character(len=4) :: a, b\n  a="foo"\n  b="bar"\n  call sub(a, b)'),
    ("dummy_slice", 'character(len=*) :: x', 'print *, x(2:3)',
     'character(len=5) :: s\n  s="hello"\n  call sub(s)'),
    ("func_arg", 'character(len=*) :: a', 'f = "ab"',
     'print *, f("hi")'),
]
CONCATS = [b for _, b in CASES]
DECLS = {b: d for d, b in CASES}

# Reference-accepted, ffc-refused: whole-array integer arithmetic.  #337
# territory, not this surface, and a compile-time refusal rather than a wrong
# answer; listed so the gap stays visible.
KNOWN_GAP: set[str] = set()


def run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    if not FFC.exists():
        print(f"SKIP: ffc not built at {FFC}", file=sys.stderr)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    runs = match = refused = mismatch = ref_fail = 0

    for i, expr in enumerate(CONCATS):
        name = f"cs{i}"
        src = WORK / f"{name}.f90"
        src.write_text(
            f"program p\n  {DECLS[expr]}\n  print *, {expr}\nend program p\n"
        )
        rcode, rr_out, rr_err = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
        if rcode != 0:
            ref_fail += 1
            lines.append(f"{name}\tREF_FAIL")
            continue
        rr = run([str(WORK / f"{name}_ref")])
        rc, _, ffc_err = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
        runs += 1
        if rr[0] != 0:
            lines.append(f"{name}\tREF_RUNTIME_FAIL")
            continue
        if rc != 0:
            tag = "REFUSED" if "unsupported" in ffc_err else "OTHER_ERROR"
            if name in KNOWN_GAP:
                lines.append(f"{name}\tKNOWN_GAP\t{tag}")
                continue
            refused += 1
            lines.append(f"{name}\t{tag}")
            continue
        fr = run([str(WORK / f"{name}_ffc")])
        rmd5 = hashlib.md5(rr[1].encode()).hexdigest()
        fmd5 = hashlib.md5(fr[1].encode()).hexdigest()
        if fr[0] != 0:
            refused += 1
            lines.append(f"{name}\tFFC_RUNTIME_FAIL\t{fmd5}")
            continue
        if rmd5 == fmd5:
            match += 1
            lines.append(f"{name}\tMATCH\t{rmd5}")
        else:
            mismatch += 1
            lines.append(f"{name}\tMISMATCH\tref={rmd5[:12]}\tffc={fmd5[:12]}")

    # Contained-subprogram forms: dummy/argument descriptor passing.
    for name, dummy_decl, body, main in DUMMY_CASES:
        src = WORK / f"{name}.f90"
        if name == "func_arg" or name == "func_result_cat":
            prog = (
                f"program p\n  {main}\ncontains\n  function f({dummy_decl.split(':: ')[1].strip()})\n"
                f"    {dummy_decl}\n    character(len=4) :: f\n    {body}\n  end function f\nend program p\n"
            )
        else:
            prog = (
                f"program p\n  {main}\ncontains\n  subroutine sub({dummy_decl.split(':: ')[1].strip()})\n"
                f"    {dummy_decl}\n    {body}\n  end subroutine sub\nend program p\n"
            )
        src.write_text(prog)
        rcode, _, _ = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
        if rcode != 0:
            ref_fail += 1
            lines.append(f"{name}\tREF_FAIL")
            continue
        rr = run([str(WORK / f"{name}_ref")])
        rc, _, ffc_err = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
        runs += 1
        if rc != 0:
            refused += 1
            lines.append(f"{name}\tREFUSED\t{ffc_err.strip()[:60]}")
            continue
        fr = run([str(WORK / f"{name}_ffc")])
        rmd5 = hashlib.md5(rr[1].encode()).hexdigest()
        fmd5 = hashlib.md5(fr[1].encode()).hexdigest()
        if rmd5 == fmd5:
            match += 1
            lines.append(f"{name}\tMATCH\t{rmd5}")
        else:
            mismatch += 1
            lines.append(f"{name}\tMISMATCH\tref={rmd5[:12]}\tffc={fmd5[:12]}")

    REPORT.write_text("\n".join(lines) + "\n")
    summary = (
        f"runs={runs} match={match} refused={refused} mismatch={mismatch} "
        f"ref_fail={ref_fail}"
    )
    print(summary)
    print(f"report={REPORT}")
    digests = {l.split("\t")[2] for l in lines if l.split("\t")[1] == "MATCH"}
    print(f"distinct_output_digests={len(digests)} rows={match}")
    return 1 if (mismatch or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
