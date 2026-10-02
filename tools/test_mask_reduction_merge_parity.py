#!/usr/bin/env python3
"""Byte-exact parity for MASK= reductions and merge (#766); pins numeric core.

Three gaps, isolated so each has its own row (#766):
  * `sum(a, mask=...)`, `maxval(a, mask=...)`, `minval(a, mask=...)` refuse with
    "requires exactly one array argument" - the call HAS exactly one array argument,
    so the message describes an arity error that does not exist. Following it removes
    the mask and silently changes 12 to 15, turning a diagnostic into a wrong answer.
  * `count(mask=...)` refuses "AST node is not an identifier" - a different root
    cause: keyword-argument resolution dies on the logical expression. `count(a>2)`
    positional works, which localises it to keyword resolution, not to `count`.
  * `merge(1,2,a>2)` unimplemented.

The numeric rows here are the ones that PASS and are kept as regression guards, not
because they are interesting: Fortran's `mod` takes the sign of the dividend
(-7 mod 2 = -1, 7 mod -2 = 1), `nint(-2.7)` = -3, `floor(-2.7)` = -3,
`int(-2.7)` = -2, int8 HUGE, and signed overflow wraparound all match today. Pinning
them means a future routing change cannot quietly break the arithmetic that already
works while adding the mask support.

KNOWN_REFUSED rows score as progress-when-refused; they flip to MATCH when #766
lands and the oracle fails if any of them ever produces a wrong answer instead.

Run:  python3 tools/test_mask_reduction_merge_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/mask")
REPORT = Path("/var/tmp/ffc-goal/perf/mask/report.tsv")

DECL = ("integer :: a(5)\n  integer :: b(5)\n  real :: r(4)\n  logical :: lm(5)\n  lm = [.true.,.true.,.false.,.true.,.true.]\n"
        "  a = [1,2,3,4,5]\n  b = [10,20,30,40,50]\n"
        "  r = [1.5,-2.5,3.0,-4.0]")

# FIXED at this commit: MASK= lands on sum/product/maxval/minval/count over
# fixed-size arrays (MASK= rows below now score MATCH and fail the oracle if
# they ever regress). Still refused by name, honestly:
KNOWN_REFUSED = [
    "abs_nested_int", "abs_nested_min",
    "merge_scalar", "merge_array",
]

# KIND/DIM stay honest refusals; a silently dropped KIND or DIM is the same
# failure mode as the dropped MASK that hid #766.
KNOWN_REFUSED = KNOWN_REFUSED + ["dim_refused", "kind_refused"]

CASES = [
    ("sum_plain", "sum(a)"), ("maxval_plain", "maxval(a)"),
    ("minval_plain", "minval(a)"), ("product_plain", "product(a)"),
    ("count_pos", "count(a>2)"), ("any_pos", "any(a>4)"),
    ("all_pos", "all(a>0)"), ("all_false", "all(a>2)"),
    # Nested ABS inside a reduction: ffc calls this an "ABS reduction expression"
    # and demands a REAL array, so `maxval(abs(a))` on INTEGER a refuses while
    # `abs(-5)` and `abs(-5.5)` both work. Fourth gap in #766.
    ("abs_nested_int", "maxval(abs(a))"),
    ("abs_nested_min", "minval(abs(a))"),
    ("abs_scalar_int", "abs(-5)"),
    ("abs_scalar_real", "abs(-5.5)"),
    ("sum_mask", "sum(a, mask=a>2)"),
    ("sum_mask_all_true", "sum(a, mask=a>0)"),
    ("sum_mask_none", "sum(a, mask=a>99)"),
    ("maxval_mask", "maxval(a, mask=a<4)"),
    ("maxval_mask_excludes_max", "maxval(a, mask=a<5)"),
    ("maxval_mask_none", "maxval(a, mask=a>99)"),
    ("minval_mask", "minval(a, mask=a<4)"),
    ("minval_mask_excludes_min", "minval(a, mask=a>1)"),
    ("minval_mask_none", "minval(a, mask=a>99)"),
    ("product_mask", "product(a, mask=a<3)"),
    ("product_mask_none", "product(a, mask=a>99)"),
    ("count_kw_mask", "count(mask=a>2)"),
    ("sum_pos_mask", "sum(a, a>2)"),
    ("sum_mask_ident", "sum(a, mask=lm)"),
    ("sum_r_mask", "sum(r, mask=r>0)"),
    ("maxval_r_mask", "maxval(r, mask=r>0)"),
    ("maxval_r_mask_none", "maxval(r, mask=r>100.0)"),
    ("minval_r_mask", "minval(r, mask=r<-1.0)"),
    ("minval_r_mask_none", "minval(r, mask=r>100.0)"),
    ("product_r_mask", "product(r, mask=r>0)"),
    ("dim_refused", "sum(a, dim=1)"),
    ("kind_refused", "maxval(a, mask=a>2, kind=8)"),
    ("merge_scalar", "merge(1,2,a>2)"),
    ("merge_array", "merge(a,b,a>2)"),
    # numeric core - passes today, pinned as regression guards
    ("div_mod_neg", "__LOCAL__"),
    ("trunc_fns", "__LOCAL__"),
    ("int8_huge", "__LOCAL__"),
    ("overflow_wrap", "__LOCAL__"),
    ("real_round", "__LOCAL__"),
    ("mod_sign", "__LOCAL__"),
]

SPECIAL = {
    "div_mod_neg": ("integer :: x,y\n  x=-7\n  y=2", "x/y, mod(x,y)"),
    "trunc_fns": ("real :: q\n  q=-2.7", "int(q), nint(q), floor(q), ceiling(q)"),
    "int8_huge": ("integer(8) :: k\n  k=9223372036854775807_8", "k"),
    "overflow_wrap": ("integer :: i\n  i=2147483647", "i+1"),
    "real_round": ("real :: pr\n  pr=1.0/3.0", "pr, pr*3.0"),
    "mod_sign": ("integer :: m,n\n  m=7\n  n=-2", "m/n, mod(m,n)"),
}


def run(c):
    p = subprocess.run(c, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout


def main() -> int:
    if not FFC.exists():
        print(f"SKIP: ffc not built at {FFC}", file=sys.stderr)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    lines, runs, match, refused = [], 0, 0, 0
    for name, expr in CASES:
        decl, body = DECL, expr
        if expr == "__LOCAL__":
            decl, body = SPECIAL[name]
        src = WORK / f"{name}.f90"
        src.write_text(f"program prog\n  implicit none\n  {decl}\n  print *, {body}\n"
                       "end program prog\n")
        grc, _ = run(["gfortran", str(src), "-o", str(WORK / f"{name}_r")])
        if grc != 0 and name in KNOWN_REFUSED:
            # Invalid on this gfortran (e.g. KIND= predates it on maxval);
            # ffc refusing the same input is agreement, not a bad row.
            frc0, ferr0 = run([str(FFC), str(src), "-o", str(WORK / f"{name}_f")])
            runs += 1
            if frc0 != 0:
                refused += 1
                lines.append(f"{name}\tKNOWN_REFUSED\tboth_refuse")
            else:
                lines.append(f"{name}\tREGRESSION\tffc_accepts_invalid")
            continue
        if grc != 0:
            lines.append(f"{name}\tBAD_CASE\tgfortran_refuses")
            continue
        frc, ferr = run([str(FFC), str(src), "-o", str(WORK / f"{name}_f")])
        runs += 1
        if frc != 0:
            if name in KNOWN_REFUSED:
                refused += 1
                lines.append(f"{name}\tKNOWN_REFUSED\t{ferr.strip()[:40]}")
            else:
                lines.append(f"{name}\tUNEXPECTED_REFUSE\t{ferr.strip()[:40]}")
            continue
        _, g = run([str(WORK / f"{name}_r")])
        _, f = run([str(WORK / f"{name}_f")])
        gm, fm = hashlib.md5(g.encode()).hexdigest(), hashlib.md5(f.encode()).hexdigest()
        if gm == fm:
            match += 1
            lines.append(f"{name}\tMATCH\t{gm}")
        else:
            lines.append(f"{name}\tREGRESSION\tg={g.strip()!r}\tffc={f.strip()!r}")
    REPORT.write_text("\n".join(lines) + "\n")
    bad = [l for l in lines if l.split("\t")[1] in ("REGRESSION", "UNEXPECTED_REFUSE",
                                                      "BAD_CASE")]
    print(f"runs={runs} match={match} known_refused={refused} unexpected={len(bad)}")
    print(f"report={REPORT}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
