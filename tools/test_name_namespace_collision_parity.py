#!/usr/bin/env python3
"""Require Fortran namespace and validation decisions to agree with gfortran.

Every pinned invalid program must be refused; accepted programs must match the
reference compiler's exit status and stdout byte for byte. Build once with
``fo build`` before running. Use --work-dir to retain isolated evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
    # Type/entity collisions in these pinned scopes must be refused.
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
    # Other invalid-program shapes recorded on fortfront#3021.
    ("enddo_label_mismatch", "refuse", 'program p\n  implicit none\n'
     '  integer :: i\n  outer: do i = 1, 3\n  end do inner\nend program p\n'),
    ("dup_decl", "refuse", 'program p\n  implicit none\n  integer :: x\n'
     '  integer :: x\n  x = 1\n  print *, x\nend program p\n'),
    ("assign_parameter", "refuse", 'program p\n  implicit none\n'
     '  integer, parameter :: k = 3\n  k = 4\n  print *, k\nend program p\n'),
    ("intrinsic_bad_arity", "refuse", 'program p\n  implicit none\n'
     '  print *, len("ab", 3, 4)\nend program p\n'),
    ("endprog_name_mismatch", "refuse", 'program p\n  implicit none\n'
     '  print *, 1\nend program q\n'),
    # Valid nearby scoping forms must continue to compile and agree on output.
    ("parameter_block_shadow", "accept", 'program p\n'
     'integer, parameter :: k=3\nblock\ninteger k\nk=4\nprint *, k\n'
     'end block\nprint *, k\nend program p\n'),
    ("parameter_dummy_shadow", "accept", 'module m\n'
     'integer, parameter :: k=3\ncontains\nsubroutine s(k)\ninteger k\n'
     'k=4\nend subroutine\nend module\nprogram p\nuse m\ninteger v\n'
     'v=1\ncall s(v)\nprint *, v\nend program p\n'),
    ("parameter_as_keyword", "accept", 'program p\n'
     'integer, parameter :: kind=4\nprint *, len("abc",kind=kind)\n'
     'end program p\n'),
    ("renamed_type_collision", "refuse", MOD + 'program p\n'
     'use ns_types, only: local => shared\ninteger local\nend program p\n'),
    ("common_implicit_type_collision", "refuse", MOD + 'program p\n'
     'use ns_types\ncommon /grp/ shared\nend program p\n'),
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


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, bytes]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=120)
    return p.returncode, p.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--fo", default="fo", help="fo executable")
    args = parser.parse_args()
    if not shutil.which(args.fo) or not shutil.which("gfortran"):
        parser.error("fo and gfortran must be installed")
    compiler = [args.fo, "exec", "--no-build", "ffc"]
    status, _ = run(compiler + ["--version"])
    if status != 0:
        parser.error("ffc must be built first with fo build")
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="ffc-nscoll-", dir="/var/tmp"))
    work = work.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    report = work / "report.tsv"
    lines: list[str] = []
    runs = agree = disagree = 0
    for name, expect, src in CASES:
        f = work / f"{name}.f90"
        f.write_text(src)
        g_ok, _ = run(["gfortran", str(f), "-o", str(work / f"{name}_r")], cwd=work)
        f_ok, _ = run([*compiler, str(f), "-o", str(work / f"{name}_f")])
        gd = "accept" if g_ok == 0 else "refuse"
        fd = "accept" if f_ok == 0 else "refuse"
        if gd != expect:
            lines.append(f"{name}\tBAD_CASE\tgfortran={gd} expected={expect}")
            runs += 1
            disagree += 1
            continue
        runs += 1
        if gd != fd:
            disagree += 1
            lines.append(f"{name}\tDECISION_DISAGREE\tgfortran={gd}\tffc={fd}")
            continue
        if fd == "refuse":
            agree += 1
            lines.append(f"{name}\tAGREE_REFUSE\tboth_refuse")
            continue
        grc, gout = run([str(work / f"{name}_r")], cwd=work)
        frc, fout = run([str(work / f"{name}_f")], cwd=work)
        gm = hashlib.md5(gout).hexdigest()
        fm = hashlib.md5(fout).hexdigest()
        if gm == fm and grc == frc:
            agree += 1
            lines.append(f"{name}\tAGREE_OUTPUT\t{gm}")
        else:
            disagree += 1
            lines.append(f"{name}\tOUTPUT_DISAGREE\tg={gm[:12]}:{gout!r}\t"
                         f"f={fm[:12]}:{fout!r}")
    report.write_text("\n".join(lines) + "\n")
    print(f"runs={runs} agree={agree} disagree={disagree} known_overaccept=0")
    print(f"report={report}")
    return 1 if disagree or runs < 20 else 0


if __name__ == "__main__":
    sys.exit(main())
