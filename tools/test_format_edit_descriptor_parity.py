#!/usr/bin/env python3
"""Byte-exact gfortran parity for Iw.m, Aw, and integer B/O/Z descriptors.

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
BOZ rows additionally pin storage-width negative patterns, overflow stars,
minimum digits, and zero-width/zero-minimum blank fields. The compiler runs
through `fo exec --no-build`; compile and runtime failures fail the oracle.

Run:  python3 tools/test_format_edit_descriptor_parity.py
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/var/tmp/ffc-goal/perf/fmted")

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

BOZ_CASES = [
    ("b_pad", "", 'print "(B8)", 5'),
    ("o_pad", "", 'print "(O5)", 15'),
    ("z_pad", "", 'print "(Z4)", 15'),
    ("b_exact", "", 'print "(B4)", 15'),
    ("o_exact", "", 'print "(O2)", 63'),
    ("z_exact", "", 'print "(Z2)", 255'),
    ("b_overflow", "", 'print "(B3)", 15'),
    ("o_overflow", "", 'print "(O1)", 8'),
    ("z_overflow", "", 'print "(Z1)", 16'),
    ("b_minimum", "", 'print "(B8.5)", 5'),
    ("o_minimum", "", 'print "(O8.5)", 15'),
    ("z_minimum", "", 'print "(Z8.5)", 15'),
    ("b_zero", "", 'print "(B4)", 0'),
    ("o_zero", "", 'print "(O4)", 0'),
    ("z_zero", "", 'print "(Z4)", 0'),
    ("boz_mixed", "", 'print "(A2,B8,1X,O5,1X,Z4)", "ok", 5, 15, 15'),
    ("boz_repeated", "", 'print "(2(Z4))", 15, 255'),
    ("boz_reversion", "", 'print "(Z4)", 15, 255, 16'),
    ("boz_write", "", 'write(*,"(B8,O5,Z4)") 5, 15, 15'),
    ("boz_lowercase", "", 'print "(b8,o5,z4)", 5, 15, 15'),
    ("z_i64_expression", "integer(8) :: n\n  n=4294967296_8", 'print "(Z0)", n+1_8'),
    ("z_i64_literal", "", 'print "(Z0)", 4294967296_8'),
    ("z_negative_literal_i8", "", 'print "(Z2)", -1_1'),
    ("z_negative_literal_i64", "", 'print "(Z16)", -1_8'),
]

for descriptor in ("B", "O", "Z"):
    for width, minimum in ((0, 0), (4, 0), (0, 5), (8, 5)):
        BOZ_CASES.append((f"{descriptor.lower()}_zero_w{width}_m{minimum}", "",
                          f'print "({descriptor}{width}.{minimum})", 0'))
    for kind, width in ((1, 8), (2, 16), (4, 32), (8, 64)):
        decl = f"integer({kind}) :: n\n  n=-1_{kind}"
        BOZ_CASES.append((f"{descriptor.lower()}_negative_kind{kind}", decl,
                          f'print "({descriptor}{width})", n'))
    BOZ_CASES.append((f"{descriptor.lower()}_negative_overflow", "",
                      f'print "({descriptor}3)", -1'))
CASES.extend(BOZ_CASES)


def run(cmd: list[str]) -> tuple[int, bytes, bytes]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, timeout=120)
    return p.returncode, p.stdout, p.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=WORK)
    work = parser.parse_args().work_dir
    work.mkdir(parents=True, exist_ok=True)
    report = work / "report.tsv"
    lines: list[str] = []
    runs = match = refused = mismatch = ref_fail = run_fail = 0

    for name, decl, stmt in CASES:
        src = work / f"{name}.f90"
        src.write_text(f"program p\n  implicit none\n  {decl}\n  {stmt}\nend program p\n")
        rc_g, _, gerr = run(["gfortran", str(src), "-o", str(work / f"{name}_ref")])
        if rc_g != 0:
            ref_fail += 1
            lines.append(f"{name}\tREF_FAIL\t{gerr.strip()[:40]}")
            continue
        rc_f, _, ferr = run(["fo", "exec", "--no-build", "ffc", str(src),
                            "-o", str(work / f"{name}_ffc")])
        if rc_f != 0:
            refused += 1
            tag = "REFUSED" if b"unsupported" in ferr else "OTHER_ERROR"
            lines.append(f"{name}\t{tag}\t{ferr.strip()[:40]}")
            continue
        rr = run([str(work / f"{name}_ref")])
        fr = run([str(work / f"{name}_ffc")])
        runs += 1
        if rr[0] or fr[0]:
            run_fail += 1
            lines.append(f"{name}\tRUN_FAIL\tref_exit={rr[0]}\tffc_exit={fr[0]}")
            continue
        rmd5 = hashlib.md5(rr[1]).hexdigest()
        fmd5 = hashlib.md5(fr[1]).hexdigest()
        if rmd5 == fmd5:
            match += 1
            lines.append(f"{name}\tMATCH\tref={rmd5}\tffc={fmd5}")
        else:
            mismatch += 1
            lines.append(
                f"{name}\tMISMATCH\tref={rmd5}:{rr[1]!r}\tffc={fmd5}:{fr[1]!r}"
            )

    report.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={match} refused={refused} mismatch={mismatch} "
          f"ref_fail={ref_fail} run_fail={run_fail}")
    print(f"report={report}")
    return 1 if (mismatch or refused or ref_fail or run_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
