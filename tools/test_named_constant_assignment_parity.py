#!/usr/bin/env python3
"""Byte-exact oracle: assignments to named constants vs gfortran (#3021).

Every invalid-on-gfortran shape must be REFUSED by ffc (never executed);
every valid shape must compile and print byte-identical output. Rows that
gfortran itself refuses are scored BAD_CASE to keep the fixtures honest.
Falsification: with the lower_assignment guard removed, the invalid rows
become acceptances and this script fails with REGRESSION rows.
"""
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFC = ROOT / "build" / "fo" / "app" / "ffc"
WORK = Path("/var/tmp/ffc-goal/perf/paramassign")

# (name, program body after the declarations, expect)
# expect: "valid" -> both compile, outputs must match byte-exactly
#         "refuse" -> gfortran rejects; ffc must reject too
CASES = [
    ("param_scalar", "program p\n implicit none\n integer, parameter :: k = 3\n k = 4\n print *, k\nend program p\n", "refuse"),
    ("param_real", "program p\n implicit none\n real, parameter :: x = 1.0\n x = 2.0\n print *, x\nend program p\n", "refuse"),
    ("param_double", "program p\n implicit none\n double precision, parameter :: y = 2.0d0\n y = 3.0d0\n print *, y\nend program p\n", "refuse"),
    ("param_integer_kind", "program p\n implicit none\n integer(8), parameter :: n = 5_8\n n = 6_8\n print *, n\nend program p\n", "refuse"),
    ("param_array_element", "program p\n implicit none\n integer, parameter :: a(3) = [1,2,3]\n a(2) = 9\n print *, a(2)\nend program p\n", "refuse"),
    ("param_array_whole", "program p\n implicit none\n integer, parameter :: a(3) = [1,2,3]\n a = [4,5,6]\n print *, a\nend program p\n", "refuse"),
    ("param_negated_value", "program p\n implicit none\n integer, parameter :: k = 3\n print *, k\n k = k + 1\n print *, k\nend program p\n", "refuse"),
    ("param_two_targets", "program p\n implicit none\n integer, parameter :: k = 1, m = 2\n k = 5\n print *, k, m\nend program p\n", "refuse"),
    ("param_second_only", "program p\n implicit none\n integer, parameter :: k = 1, m = 2\n m = 5\n print *, k, m\nend program p\n", "refuse"),
    ("param_in_contained", "program p\n implicit none\n integer, parameter :: k = 3\n call work\ncontains\n subroutine work\n k = 7\n print *, k\n end subroutine\nend program p\n", "refuse"),
    ("param_plus_self_rhs", "program p\n implicit none\n integer :: s\n integer, parameter :: k = 2\n s = k + k\n print *, s\nend program p\n", "valid"),
    ("shadow_other_unit", "program p\n implicit none\n integer :: k\n k = 4\n print *, k\nend program p\n", "valid"),
    ("local_same_name", "program p\n implicit none\n call work\ncontains\n subroutine work\n integer :: k\n k = 9\n print *, k\n end subroutine\nend program p\n", "valid"),
    ("param_then_var_array", "program p\n implicit none\n integer, parameter :: n = 3\n integer :: a(n)\n a = 7\n print *, a\nend program p\n", "valid"),
    ("param_readonly_prints", "program p\n implicit none\n integer, parameter :: k = 3\n print *, k\nend program p\n", "valid"),
    ("param_bounds_loop", "program p\n implicit none\n integer, parameter :: n = 3\n integer :: i, s\n s = 0\n do i = 1, n\n s = s + i\n end do\n print *, s\nend program p\n", "valid"),
    ("var_reassigned_valid", "program p\n implicit none\n integer :: k\n k = 3\n k = 4\n print *, k\nend program p\n", "valid"),
    ("param_char_assign", "program p\n implicit none\n character(len=3), parameter :: c = 'abc'\n c = 'xyz'\n print *, c\nend program p\n", "refuse"),
    ("param_logical", "program p\n implicit none\n logical, parameter :: l = .true.\n l = .false.\n print *, l\nend program p\n", "refuse"),
    ("array_var_element_valid", "program p\n implicit none\n integer :: a(3)\n a(2) = 9\n print *, a(2)\nend program p\n", "valid"),
    ("param_in_if_rhs", "program p\n implicit none\n integer, parameter :: n = 2\n integer :: s\n s = 0\n if (n > 1) s = n * 3\n print *, s\nend program p\n", "valid"),
    ("param_self_rhs_valid", "program p\n implicit none\n integer, parameter :: k = 5\n integer :: t\nt = k\n t = t + k\n print *, t\nend program p\n", "valid"),
    ("shadow_inner_scope_ok", "program p\n implicit none\n integer, parameter :: k = 3\n block\n integer :: k2\n k2 = k + 1\n print *, k2\n end block\nend program p\n", "valid"),
    ("param_complex", "program p\n implicit none\n complex, parameter :: z = (1.0,2.0)\n z = (3.0,4.0)\n print *, z\nend program p\n", "refuse"),
    ("param_after_valid_prefix", "program p\n implicit none\n integer :: s\n integer, parameter :: k = 3\n s = 1\n k = 2\n print *, s\nend program p\n", "refuse"),
]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout + p.stderr


def main():
    if not FFC.exists():
        print(f"missing ffc binary: {FFC}")
        return 2
    WORK.mkdir(parents=True, exist_ok=True)
    lines = []
    runs = matches = refused = bad = regressions = 0
    for name, src, expect in CASES:
        f = WORK / f"{name}.f90"
        f.write_text(src)
        grc, gerr = run(["gfortran", str(f), "-o", str(WORK / f"{name}_g")])
        if expect == "valid" and grc != 0:
            lines.append(f"{name}\tBAD_CASE\tgfortran_refuses_valid")
            bad += 1
            continue
        if expect == "refuse" and grc == 0:
            lines.append(f"{name}\tBAD_CASE\tgfortran_accepted")
            bad += 1
            continue
        frc, ferr = run([str(FFC), str(f), "-o", str(WORK / f"{name}_f")])
        runs += 1
        if expect == "refuse":
            if frc != 0:
                refused += 1
                lines.append(f"{name}\tREFUSED_AGREED\t")
            else:
                rc, out = run([str(WORK / f"{name}_f")])
                lines.append(f"{name}\tREGRESSION\tffc_accepts_ran_out={out.strip()[:24]!r}")
                regressions += 1
            continue
        if frc != 0:
            lines.append(f"{name}\tREGRESSION\tvalid_refused:{ferr.strip()[:40]!r}")
            regressions += 1
            continue
        _, g = run([str(WORK / f"{name}_g")])
        _, o = run([str(WORK / f"{name}_f")])
        gm, om = (hashlib.md5(g.encode()).hexdigest(),
                  hashlib.md5(o.encode()).hexdigest())
        if gm == om:
            matches += 1
            lines.append(f"{name}\tMATCH\t{gm}")
        else:
            lines.append(f"{name}\tREGRESSION\tg={g.strip()!r}\tffc={o.strip()!r}")
            regressions += 1
    (WORK / "report.tsv").write_text("\n".join(lines) + "\n")
    print(f"runs={runs} match={matches} refused_agreed={refused} "
          f"regressions={regressions} bad={bad}")
    print(f"report={WORK / 'report.tsv'}")
    return 1 if regressions or bad else 0


if __name__ == "__main__":
    sys.exit(main())
