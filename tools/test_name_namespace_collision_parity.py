#!/usr/bin/env python3
"""Pin name-namespace collision behaviour against gfortran; records live over-acceptance.

PLAN W1.e carried "symbol-collision false positives (`Symbol 't' names an
incompatible object`)" forward from a prior plan. The direction was wrong in the
plan AND in my first reading of it. Measured against gfortran, the project's
behavioral oracle:

- fortfront's `semantic_local_name_collision_validation.f90` check fires on a DO
  **construct label** that repeats a host/use-associated derived type name, and
  gfortran refuses the same program at that line -> **AGREE**, that half is right.
- What is missing is the **variable / loop-variable / COMMON-member** half. Four
  shapes are invalid Fortran that gfortran rejects and ffc compiles and runs:

      integer :: shared      with `type :: shared` use-associated
          gfortran: Symbol 'shared' also declared as a type
          ffc:      compiles, prints            9
      common /g/ shared      member repeating a use-associated type name
          gfortran: same diagnostic             ffc: compiles, prints  2
      do shared = 1, 2       loop variable named after a derived type
          gfortran: Derived type 'shared' cannot be used as a variable
          ffc:      compiles, prints            2

  That is ffc silently accepting invalid Fortran - a soundness gap, which is the
  opposite failure mode from the one the plan recorded, and the more serious one:
  the rejection gate exists to keep this set from shrinking by accident.

Oracle shape: for each case both compilers' DECISIONS must agree (accept/refuse),
and when both accept, outputs must match byte for byte. The four over-accepted
programs are `KNOWN_OVERACCEPT`, counted and named in the report so the guard stays
usable as a regression net while the count of silently-accepted invalid programs
stays visible and can only fall when the check lands.

Run:  python3 tools/test_name_namespace_collision_parity.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/nscoll")
REPORT = Path("/var/tmp/ffc-goal/perf/nscoll/report.tsv")

MOD = (
    "module ns_types\n  implicit none\n"
    "  type :: bucket\n    integer :: count\n  end type bucket\n"
    "  type :: shared\n    integer :: n\n  end type shared\n"
    "end module ns_types\n"
)

# (name, expect, source)  expect in {"accept", "refuse"}
CASES = [
    # distinct names in every namespace -> valid, must agree on output
    ("distinct_all", "accept", MOD + 'program p\n  use ns_types\n  implicit none\n'
     '  type(bucket) :: o\n  integer :: i\n  o%count = 0\n'
     '  outer: do i = 1, 3\n    o%count = o%count + i\n  end do outer\n'
     '  print *, o%count\nend program p\n'),
    ("plain_do_label", "accept", 'program p\n  implicit none\n  integer :: i, s\n'
     '  s = 0\n  loop: do i = 1, 4\n    s = s + i\n  end do loop\n'
     '  print *, s\nend program p\n'),
    ("common_distinct", "accept", MOD + 'program p\n  use ns_types\n  implicit none\n'
     '  integer :: total\n  common /grp/ total\n  type(shared) :: obj\n'
     '  total = 5\n  obj%n = 6\n  print *, total, obj%n\nend program p\n'),
    ("local_shadow_ok", "accept", 'program p\n  implicit none\n  integer :: x\n'
     '  x = 1\n  block\n    integer :: x\n    x = 2\n    print *, x\n  end block\n'
     '  print *, x\nend program p\n'),
    ("type_and_var_distinct", "accept", MOD + 'program p\n  use ns_types\n'
     '  implicit none\n  type(bucket) :: b\n  integer :: k\n'
     '  b%count = 2\n  k = 3\n  print *, b%count + k\nend program p\n'),
    ("do_label_no_type_same", "accept", 'program p\n  implicit none\n'
     '  integer :: i, t\n  t = 0\n  cyc: do i = 1, 2\n    t = t + i\n  end do cyc\n'
     '  print *, t\nend program p\n'),
    ("associate_name_ok", "accept", MOD + 'program p\n  use ns_types\n'
     '  implicit none\n  type(bucket) :: o\n  o%count = 4\n'
     '  associate(c => o%count)\n    print *, c\n  end associate\n'
     'end program p\n'),
    ("nested_labels_ok", "accept", 'program p\n  implicit none\n'
     '  integer :: i, j, s\n  s = 0\n  a: do i = 1, 2\n    b: do j = 1, 2\n'
     '      s = s + 1\n    end do b\n  end do a\n  print *, s\nend program p\n'),
    # collisions with a host/use-associated TYPE name -> both must refuse
    ("var_collides_type", "refuse", MOD + 'program p\n  use ns_types\n'
     '  implicit none\n  integer :: shared\n  shared = 9\n'
     '  print *, shared\nend program p\n'),
    ("do_label_collides_type", "refuse", MOD + 'program p\n  use ns_types\n'
     '  implicit none\n  integer :: i\n  type(bucket) :: o\n  o%count = 0\n'
     '  bucket: do i = 1, 3\n    o%count = o%count + 1\n  end do bucket\n'
     '  print *, o%count\nend program p\n'),
    ("var_collides_bucket", "refuse", MOD + 'program p\n  use ns_types\n'
     '  implicit none\n  integer :: bucket\n  bucket = 1\n'
     '  print *, bucket\nend program p\n'),
    ("common_member_collides_type", "refuse", MOD + 'program p\n  use ns_types\n'
     '  implicit none\n  integer :: shared\n  common /g/ shared\n'
     '  shared = 2\n  print *, shared\nend program p\n'),
    ("loopvar_collides_type", "refuse", MOD + 'program p\n  use ns_types\n'
     '  implicit none\n  type(bucket) :: o\n  o%count = 0\n'
     '  do shared = 1, 2\n    o%count = o%count + 1\n  end do\n'
     '  print *, o%count\nend program p\n'),
    # plain valid programs, unrelated namespaces, must agree on output
    ("scalar_arith", "accept", 'program p\n  implicit none\n'
     '  print *, 2+3, 7-1, 3*4, 8/2\nend program p\n'),
    ("four_sign_div", "accept", 'program p\n  implicit none\n'
     '  print *, 7/2, -7/2, 7/(-2), (-7)/(-2)\nend program p\n'),
    ("mod_signs", "accept", 'program p\n  implicit none\n'
     '  print *, mod(7,2), mod(-7,2), mod(7,-2)\nend program p\n'),
    ("char_slice", "accept", 'program p\n  implicit none\n'
     '  character(len=6) :: s\n  s="abcdef"\n  print *, s(2:4), len_trim(s)\n'
     'end program p\n'),
    ("trim_concat", "accept", 'program p\n  implicit none\n'
     '  character(len=8) :: s\n  s="ab"\n  print *, trim(s)//"Z"\nend program p\n'),
    ("cmp_logical", "accept", 'program p\n  implicit none\n'
     '  print *, 1<2, 3>4, 2==2\nend program p\n'),
    ("array_sum", "accept", 'program p\n  implicit none\n  integer :: a(3)\n'
     '  a=[1,2,3]\n  print *, sum(a), maxval(a), minval(a)\nend program p\n'),
    ("implied_do", "accept", 'program p\n  implicit none\n  integer :: i, a(3)\n'
     '  a=[4,5,6]\n  print *, (a(i), i=1,3)\nend program p\n'),
    ("two_arrays", "accept", 'program p\n  implicit none\n  integer :: a(2), b(2)\n'
     '  a=[1,2]\n  b=[3,4]\n  print *, sum(a), sum(b)\nend program p\n'),
    ("real_fmt", "accept", 'program p\n  implicit none\n'
     '  print "(F8.2)", 3.5\nend program p\n'),
    ("i_zero_pad", "accept", 'program p\n  implicit none\n'
     '  print "(I5.3)", 42\nend program p\n'),
]


# Over-acceptance recorded by name: gfortran REFUSES these as invalid Fortran and
# ffc compiles them and runs them, printing a value. Counted separately so the
# guard stays green as a regression net while the number of silently-accepted
# invalid programs stays visible and can only fall when the check lands.
#   var_collides_type / var_collides_bucket  gfortran: "Symbol 'X' also declared
#     as a type"
#   common_member_collides_type                same rule via a COMMON member
#   loopvar_collides_type            gfortran: "Derived type 'X' cannot be used as
#     a variable" - ffc happily makes it a loop variable
KNOWN_OVERACCEPT = ["var_collides_type", "var_collides_bucket",
                    "common_member_collides_type", "loopvar_collides_type"]


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout


def main() -> int:
    if not FFC.exists():
        print(f"SKIP: ffc not built at {FFC}", file=sys.stderr)
        return 0
    WORK.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    runs = agree = disagree = 0
    for name, expect, src in CASES:
        f = WORK / f"{name}.f90"
        f.write_text(src)
        g_ok, _ = run(["gfortran", str(f), "-o", str(WORK / f"{name}_r")])
        f_ok, _ = run([str(FFC), str(f), "-o", str(WORK / f"{name}_f")])
        gd = "accept" if g_ok == 0 else "refuse"
        fd = "accept" if f_ok == 0 else "refuse"
        if gd == "refuse":
            # gfortran is the oracle: a program it rejects is invalid, so the
            # expected verdict was wrong if we asked for "accept".
            if expect != "refuse":
                lines.append(f"{name}\tBAD_CASE\tgfortran_refuses_but_expected_accept")
                runs += 1
                disagree += 1
                continue
        runs += 1
        if gd != fd:
            if name in KNOWN_OVERACCEPT and gd == "refuse" and fd == "accept":
                agree += 1
                lines.append(f"{name}\tKNOWN_OVERACCEPT\tffc_accepts_invalid")
                continue
            disagree += 1
            lines.append(f"{name}\tDECISION_DISAGREE\tgfortran={gd}\tffc={fd}")
            continue
        if fd == "refuse":
            agree += 1
            lines.append(f"{name}\tAGREE_REFUSE\tboth_refuse")
            continue
        grc, gout = run([str(WORK / f"{name}_r")])
        frc, fout = run([str(WORK / f"{name}_f")])
        gm = hashlib.md5(gout.encode()).hexdigest()
        fm = hashlib.md5(fout.encode()).hexdigest()
        if gm == fm and grc == frc:
            agree += 1
            lines.append(f"{name}\tAGREE_OUTPUT\t{gm}")
        else:
            disagree += 1
            lines.append(f"{name}\tOUTPUT_DISAGREE\tg={gm[:12]}:{gout!r}\t"
                         f"f={fm[:12]}:{fout!r}")
    REPORT.write_text("\n".join(lines) + "\n")
    over = sum(1 for l in lines if l.split("\t")[1] == "KNOWN_OVERACCEPT")
    print(f"runs={runs} agree={agree} disagree={disagree} known_overaccept={over}")
    print(f"report={REPORT}")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
