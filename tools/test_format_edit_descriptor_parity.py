#!/usr/bin/env python3
"""Byte-exact parity for FORMAT edit descriptors (Iw.m zero-pad, Aw truncate).

`I5.3` printed `   42` instead of `  042` and `(A3)` printed `ABCDEF` instead of
`ABC`: the print lowering parsed the width and then **discarded** the `Iw.m`
minimum-digit count (the code said so in a comment) and emitted `%Ns`, which has
no truncation. Both are silent wrong answers in ordinary formatted output.

printf already expresses exactly what Fortran means here, which is what makes the
fix small: `%w.md` is `Iw.m` (at least m digits, right-justified in w) and
`%w.Ns` is `Aw` (truncate to w, pad to w). The padding direction was verified
against gfortran rather than assumed - Fortran text says A is left-justified,
but gfortran right-justifies `(A5)`/"AB" to `   AB`, and ffc already matched that,
so only the precision field was missing.

Each row compiles under both compilers and the outputs are compared byte for
byte; the reference and ffc digests are recorded per row so a future change that
alters spacing is visible even when a test still "passes" by trailing-blank luck.
Rows gfortran itself refuses are `REF_FAIL` (reported, skipped, never a pass).

Run:  python3 tools/test_format_edit_descriptor_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/fmted")
REPORT = Path("/var/tmp/ffc-goal/perf/fmted/report.tsv")

# (name, declarations, print statement)
CASES = [
    ("i5_3_pos", "", 'print "(I5.3)", 42'),
    ("i5_3_neg", "", 'print "(I5.3)", -42'),
    ("i4_4_small", "", 'print "(I4.4)", 7'),
    ("i3_3_zero", "", 'print "(I3.3)", 0'),
    ("i6_4_neg1", "", 'print "(I6.4)", -1'),
    ("i6_3_zero", "", 'print "(I6.3)", 0'),
    ("i8_plain", "", 'print "(I8)", -42'),
    ("i0_plain", "", 'print "(I0)", 42'),
    ("i_wide_pad", "", 'print "(I10.5)", 1234'),
    ("i_exact", "", 'print "(I3.3)", 123'),
    ("a_trunc6", "", 'print "(A3)", "ABCDEF"'),
    ("a_trunc4", "", 'print "(A2)", "HIJK"'),
    ("a_pad", "", 'print "(A5)", "AB"'),
    ("a_exact", "", 'print "(A)", "FULL"'),
    ("a_pad_then_int", "", 'print "(A5,I2)", "AB", 7'),
    ("a_trunc_then_int", "", 'print "(A3,I2)", "ABCDEF", 7'),
    ("mixed_a_i", "", 'print "(A2,I4.2)", "XY", 3'),
    ("f_f10_3", "", 'print "(F10.3)", 3.14159'),
    ("f_neg", "", 'print "(F8.2)", -2.5'),
    ("e12_5", "", 'print "(E12.5)", 1.23456789'),
    ("es_small", "", 'print "(ES12.5)", 0.0001234'),
    ("l_true", "", "print \"(L1)\", .true."),
    ("l_false_w3", "", "print \"(L3)\", .false."),
    ("multi_i", "", 'print "(I2.2,I3.1)", 5, 42'),
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

    for name, decl, stmt in CASES:
        src = WORK / f"{name}.f90"
        src.write_text(f"program p\n  implicit none\n  {decl}\n  {stmt}\nend program p\n")
        rc_g, _, gerr = run(["gfortran", str(src), "-o", str(WORK / f"{name}_ref")])
        if rc_g != 0:
            ref_fail += 1
            lines.append(f"{name}\tREF_FAIL\t{gerr.strip()[:40]}")
            continue
        rc_f, _, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_ffc")])
        if rc_f != 0:
            refused += 1
            tag = "REFUSED" if "unsupported" in ferr else "OTHER_ERROR"
            lines.append(f"{name}\t{tag}\t{ferr.strip()[:40]}")
            continue
        rr = run([str(WORK / f"{name}_ref")])
        fr = run([str(WORK / f"{name}_ffc")])
        runs += 1
        rmd5 = hashlib.md5(rr[1].encode()).hexdigest()
        fmd5 = hashlib.md5(fr[1].encode()).hexdigest()
        if rmd5 == fmd5:
            match += 1
            lines.append(f"{name}\tMATCH\tref={rmd5[:12]}\tffc={fmd5[:12]}")
        else:
            mismatch += 1
            lines.append(
                f"{name}\tMISMATCH\tref={rmd5[:12]}:{rr[1]!r}\tffc={fmd5[:12]}:{fr[1]!r}"
            )

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} refused={refused} mismatch={mismatch} "
          f"ref_fail={ref_fail}")
    print(f"report={REPORT}")
    return 1 if (mismatch or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
