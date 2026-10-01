#!/usr/bin/env python3
"""Regression guard for reductions over assumed-shape dummies (ffc#756).

#756 reported `sum(x(2:3))` over an assumed-shape dummy printing **0**: the
section extent was not routed, so the reduction saw no elements. Probing the
issue repro on the current build shows the symptom is **gone** - it prints 5
(2+3), matching gfortran. Before closing a defect on a single probe this oracle
pins the whole surrounding surface, because "one case works" is weak evidence:
lower bounds, non-unit strides, full-array and 2D assumed-shape reductions all
travel through the same extent-routing path (`read_runtime_dim_extent`), and a
fix that handles only the reported shape would pass the repro while leaving the
family broken.

Every row is byte-exact against gfortran. A row gfortran itself refuses is
`REF_FAIL` - reported and skipped, never counted as a pass - and a row ffc
refuses while gfortran accepts is `KNOWN_GAP`, kept visible without turning the
guard red.

Run:  python3 tools/test_assumed_shape_reduction_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/ashred")
REPORT = Path("/var/tmp/ffc-goal/perf/ashred/report.tsv")

# (name, main_decls, call, dummy_decls, body)
CASES = [
    ("sec_2_3", "integer :: a(5)\n  a=[1,2,3,4,5]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(2:3))"),
    ("sec_2_4", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(2:4))"),
    ("full", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x)"),
    ("singleton", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(3:3))"),
    ("whole_exact", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(1:5))"),
    ("stride_odd", "integer :: a(6)\n  a=[1,2,3,4,5,6]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(1:6:2))"),
    ("stride_even", "integer :: a(6)\n  a=[1,2,3,4,5,6]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(2:6:2))"),
    ("stride_three", "integer :: a(9)\n  a=[1,2,3,4,5,6,7,8,9]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(1:9:3))"),
    ("two_stride_same", "integer :: a(6)\n  a=[1,2,3,4,5,6]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(1:6:2)), sum(x(2:6:2))"),
    ("mixed_three", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(2:4)), sum(x(1:5)), sum(x(3:3))"),
    ("size_too", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, size(x), sum(x)"),
    ("prod", "integer :: a(4)\n  a=[1,2,3,4]", "call sub(a)",
     "integer :: x(:)", "print *, product(x(2:3))"),
    ("maxval", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, maxval(x(2:4))"),
    ("minval", "integer :: a(5)\n  a=[10,20,30,40,50]", "call sub(a)",
     "integer :: x(:)", "print *, minval(x(2:4))"),
    ("sec_of_full", "integer :: a(5)\n  a=[1,2,3,4,5]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(2:3)) + sum(x(4:5))"),
    ("real_sec", "real :: a(4)\n  a=[1.0,2.0,3.0,4.0]", "call sub(a)",
     "real :: x(:)", "print *, sum(x(2:3))"),
    ("two_d_full", "integer :: a(2,3)\n  a=reshape([1,2,3,4,5,6],[2,3])", "call sub(a)",
     "integer :: x(:,:)", "print *, sum(x)"),
    ("two_d_row", "integer :: a(2,3)\n  a=reshape([1,2,3,4,5,6],[2,3])", "call sub(a)",
     "integer :: x(:,:)", "print *, sum(x(1,:))"),
    ("two_d_col", "integer :: a(2,3)\n  a=reshape([1,2,3,4,5,6],[2,3])", "call sub(a)",
     "integer :: x(:,:)", "print *, sum(x(:,2))"),
    ("two_d_block", "integer :: a(3,3)\n  a=reshape([1,2,3,4,5,6,7,8,9],[3,3])",
     "call sub(a)", "integer :: x(:,:)", "print *, sum(x(1:2,2:3))"),
    ("len6_all", "integer :: a(6)\n  a=[1,2,3,4,5,6]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x)"),
    ("len3_mid", "integer :: a(3)\n  a=[7,8,9]", "call sub(a)",
     "integer :: x(:)", "print *, sum(x(2:2))"),
]


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

    for name, decls, call, ddecl, body in CASES:
        src = WORK / f"{name}.f90"
        src.write_text(
            "program p\n"
            f"  {decls}\n"
            f"  {call}\n"
            "contains\n"
            "  subroutine sub(x)\n"
            f"    {ddecl}\n"
            f"    {body}\n"
            "  end subroutine sub\n"
            "end program p\n"
        )
        rcode, _, rerr = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
        if rcode != 0:
            ref_fail += 1
            lines.append(f"{name}\tREF_FAIL\t{rerr.strip()[:50]}")
            continue
        rc, _, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
        runs += 1
        if rc != 0:
            tag = "REFUSED" if "unsupported" in ferr else "OTHER_ERROR"
            lines.append(f"{name}\t{tag}\t{ferr.strip()[:50]}")
            refused += 1
            continue
        rr = run([str(WORK / f"{name}_ref")])
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
    print(f"runs={runs} match={match} refused={refused} mismatch={mismatch} "
          f"ref_fail={ref_fail}")
    digests = {l.split("\t")[2] for l in lines if l.split("\t")[1] == "MATCH"}
    print(f"distinct_output_digests={len(digests)} rows={match}")
    print(f"report={REPORT}")
    return 1 if (mismatch or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
