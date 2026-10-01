#!/usr/bin/env python3
"""Convert standalone wrapper test programs into cases of the dispatcher.

STATUS: not wired into the build yet - see fo issue "dispatcher routing does
not fire on the test build path". The generated dispatcher compiles and runs
(`build/fo/bin/test_ffc_suite` prints its usage and exits 2 when run bare),
but `fo test <name>` still executes the test's own binary: hiding
`build/fo/bin/test_session_empty_program_compiler` makes that test FAIL, so
routing is provably not happening. Until fo routes, this tool's output is not
applied to `test/` - a `! fo: dispatcher` marker in the tree would promise a
routing that does not occur. Run with `--apply` only together with that fix.

Each `test/test_x.f90` is its own executable, and every one of them re-links
~15 MB of libffc: `build/fo/obj` is 38 MB while `build/fo/bin` is 7.7 GB.
A case runs inside the dispatcher as `ffc_suite test_x`, so the library is
linked once. The case name is the old test name, unchanged, so `fo test
test_x` and the per-name suite report keep working.

The transformation is deliberately small: `program test_x` becomes
`subroutine case_test_x()` and the body is copied verbatim, because `stop 1`
inside a case exits the dispatcher process with the same status the standalone
test returned - a failure still looks like a failure to `fo`.

A wrapper whose helpers live inside itself (an internal subroutine or function)
is refused: those helpers would become module procedures of a different scope,
which is a real edit, not a fold. The operator does that file by hand.

Usage:
    python3 tools/make_suite_cases.py [--apply] [test/test_x.f90 ...]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

MARKER = "! fo: dispatcher"
CASES_DIR = pathlib.Path("test/suite")
MAIN = pathlib.Path("test/test_ffc_suite.f90")
INTERNAL = re.compile(
    r"^\s*(?:recursive\s+|pure\s+|elemental\s+)*"
    r"(?:integer|logical|real|complex|character|type|class)?\s*"
    r"(?:subroutine|function)\s+[A-Za-z_]", re.I)


def convert(path: pathlib.Path) -> tuple[str, str] | None:
    text = path.read_text()
    lines = text.split("\n")
    name = path.stem
    body: list[str] = []
    use: list[str] = []
    started = False
    # Fold continuations first. A `use m, only: a, &` list that stops at the
    # first physical line is a syntax error, and half the wrappers here wrap
    # their `use` lists, so the fold is not a formatting nicety.
    folded: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        while cur.rstrip().endswith("&") and i + 1 < len(lines):
            i += 1
            nxt = lines[i].strip()
            cut = cur.rstrip()[:-1].rstrip()
            if not cut.lstrip().startswith("!"):
                cur = cut + " " + nxt
            else:
                cur = cut + nxt
        folded.append(cur)
        i += 1
    lines = folded

    for l in lines:
        if not started:
            # The program statement's own name is NOT necessarily the file
            # stem - `test_session_pointer_associated2_compiler.f90` holds
            # `program test_session_pointer_associated2`. The dispatch name is
            # the file stem, because that is the name `fo` discovers and
            # reports, so match any program name here.
            if re.match(r"^\s*program\s+[A-Za-z_][A-Za-z0-9_]*\s*$", l, re.I):
                started = True
            continue
        if re.match(r"^\s*end\s+program", l, re.I):
            break
        if re.match(r"^\s*use\b", l, re.I):
            use.append(l.strip())
            continue
        if re.match(r"^\s*implicit\b", l, re.I):
            continue
        if INTERNAL.match(l) and not l.strip().startswith("!"):
            print(f"{path}: internal procedure {l.strip()[:48]} - needs a real "
                  "edit to module scope, not a fold", file=sys.stderr)
            return None
        body.append(l)
    if not started or not body:
        print(f"{path}: no wrapper program body found", file=sys.stderr)
        return None

    case = [f"    subroutine case_{name}()"]
    case += [f"        {u}" for u in use]
    case.append("        implicit none")
    case += body
    case.append(f"    end subroutine case_{name}")
    case.append("")
    return name, "\n".join(case)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("tests", nargs="*")
    a = ap.parse_args()

    targets = [pathlib.Path(t) for t in a.tests] if a.tests else sorted(
        p for p in pathlib.Path("test").glob("test_*.f90")
        if len(p.read_text().split("\n")) <= 25)

    cases: list[tuple[str, str]] = []
    skipped: list[str] = []
    for t in targets:
        r = convert(t)
        if r is None:
            # A refusal is a fact about that file, not a reason to abandon the
            # rest of the family; the operator sees the count and the names.
            skipped.append(t.name)
            continue
        cases.append(r)
    if skipped:
        print(f"skipped {len(skipped)} (internal helpers need real edits): "
              + ", ".join(skipped)[:300])

    print(f"{len(cases)} cases: " + ", ".join(n for n, _ in cases)[:400])
    if not a.apply:
        return 0

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    for chunk in range(0, len(cases), 8):
        group = cases[chunk:chunk + 8]
        mod = CASES_DIR / f"ffc_suite_cases_{chunk // 8:02d}.f90"
        out = [f"module ffc_suite_cases_{chunk // 8:02d}",
               "    !! Dispatcher cases, folded from the standalone wrapper test",
               "    !! programs of the same names. One link of libffc instead of",
               "    !! one per test; the case name is the old test name.",
               "    implicit none",
               "    public :: " + ", ".join(f"case_{n}" for n, _ in group),
               "contains", ""]
        for _, c in group:
            out.append(c)
        out.append(f"end module ffc_suite_cases_{chunk // 8:02d}")
        mod.write_text("\n".join(out) + "\n")

    # The dispatcher: `ffc_suite <case_name>` runs one case and exits with its
    # status. An unknown name is a hard error, never a silent pass - a suite
    # that reports success for a case it never ran is worse than one that
    # fails, because it hides the missing check.
    mods = [f"    use ffc_suite_cases_{i // 8:02d}, only: " +
            ", ".join(f"case_{n}" for n, _ in cases[i:i + 8])
            for i in range(0, len(cases), 8)]
    sel = []
    for n, _ in cases:
        sel.append(f'        if (argv(1) == "{n}") then')
        sel.append(f"            call case_{n}()")
        sel.append("            return")
        sel.append("        end if")
    main = ["program test_ffc_suite",
            "    !! Consolidated test binary (W0.3). `ffc_suite <test_name>`",
            "    !! runs that one case and exits with its status, so the suite",
            "    !! links libffc once instead of once per test. `fo test",
            "    !! <test_name>` routes here for every source marked",
            "    !! `! fo: dispatcher`; the name it reports is unchanged.",
            ]
    # `use` must precede `implicit none` - gfortran rejects the reverse.
    main += mods + ["    implicit none"]
    main += ["    !! An assumed-length allocatable array is not legal Fortran; the length",
             "    !! has to be explicit.",
             "    character(len=256), allocatable :: argv(:)",
             "    integer :: i",
             "",
             "    allocate(character(len=256) :: argv(max(1, command_argument_count())))",
             "    do i = 1, command_argument_count()",
             "        call get_command_argument(i, argv(i))",
             "    end do",
             "    if (command_argument_count() < 1) then",
             '        print *, "usage: ffc_suite <test_name>"',
             "        stop 2",
             "    end if",
             "    argv(1) = adjustl(argv(1))", ""]
    main += sel
    main += ["    ! Unknown name: fail loudly, do not report a pass.",
             '    print *, "ffc_suite: no such case: "//trim(argv(1))',
             "    stop 3",
             "end program test_ffc_suite"]
    MAIN.write_text("\n".join(main) + "\n")

    # Mark the originals so `fo` routes them and does not link a private copy.
    for t in targets:
        txt = t.read_text()
        if not txt.startswith(MARKER):
            t.write_text(MARKER + "\n" + txt)
    print(f"marked {len(targets)} test sources; wrote case modules in {CASES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
