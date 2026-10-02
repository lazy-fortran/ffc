#!/usr/bin/env python3
"""Consolidate test programs without copying or hoisting their test bodies.

Each original source gains a small module declaring its case procedure's
interface. Its test name remains its filename, discovered by fo through
``! fo: dispatcher``. The original program body becomes an external
subroutine, with its internal procedures and indentation preserved.
The dispatcher imports those procedures and invokes exactly one per process,
so STOP, ERROR STOP, host association and independent case execution survive.
An explicit SAVE retains the original PROGRAM's local-variable lifetime.

Run without arguments to register every test_*.f90 source; --apply performs
the conversion. Repeated runs rebuild the registry from the original sources,
including cases already converted. Test bodies have one authoritative copy.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

MARKER = "! fo: dispatcher"
MAIN = pathlib.Path("test/test_ffc_suite.f90")
GROUPS = pathlib.Path("test/suite")
GROUP_SIZE = 32
PROGRAM = re.compile(r"^\s*program\s+([a-z_]\w*)\s*$", re.I)
END_PROGRAM = re.compile(r"^\s*end\s+program(?:\s+[a-z_]\w*)?\s*$", re.I)
MODULE = re.compile(r"^module\s+([a-z_]\w*)\s*$", re.I)


def bounded(prefix: str, name: str) -> str:
    candidate = prefix + name
    if len(candidate) <= 63:
        return candidate
    return prefix + hashlib.sha256(name.encode()).hexdigest()[:24]


def module_name(name: str) -> str:
    return bounded("ffc_case_", name)


def sym(name: str) -> str:
    return bounded("case_", name)


def converted_module(path: pathlib.Path) -> str | None:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != MARKER:
        return None
    for line in lines[1:]:
        match = MODULE.match(line)
        if match:
            return match[1]
        if line.strip() and not line.lstrip().startswith("!"):
            return None
    return None


def convert(path: pathlib.Path) -> str:
    """Keep every original procedure in its original lexical scope."""
    existing = converted_module(path)
    if existing:
        if existing.lower() != module_name(path.stem).lower():
            raise ValueError(f"{path}: unexpected case module {existing}")
        return path.read_text()

    lines = path.read_text().splitlines()
    starts = [(i, PROGRAM.match(line)) for i, line in enumerate(lines)
              if PROGRAM.match(line)]
    ends = [i for i, line in enumerate(lines) if END_PROGRAM.match(line)]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"{path}: needs exactly one standalone program")
    start, _ = starts[0]
    end = ends[0]
    if end <= start:
        raise ValueError(f"{path}: invalid program boundaries")
    if any(line.strip() and not line.lstrip().startswith("!")
           for line in lines[end + 1:]):
        raise ValueError(f"{path}: external units need an explicit conversion")

    body = lines[start + 1:end]
    implicit = next((i for i, line in enumerate(body)
                     if re.match(r"^\s*implicit\s+none\b", line, re.I)), None)
    if implicit is None:
        raise ValueError(f"{path}: explicit IMPLICIT NONE required")
    body.insert(implicit + 1, "    save")
    module = module_name(path.stem)
    procedure = sym(path.stem)
    out = [MARKER, *lines[:start], f"module {module}",
           "    implicit none", "    private", f"    public :: {procedure}",
           "    interface", f"        subroutine {procedure}()",
           f"        end subroutine {procedure}", "    end interface",
           f"end module {module}", "", f"subroutine {procedure}()"]
    out.extend(body)
    out.extend([f"end subroutine {procedure}", ""])
    return "\n".join(out)


def group_source(index: int, paths: list[pathlib.Path]) -> str:
    module = f"ffc_suite_group_{index:02d}"
    out = [f"module {module}"]
    for path in paths:
        alias = sym(path.stem)
        out.extend([f"    use {module_name(path.stem)}, only: &",
                    f"        {alias}"])
    out.extend(["    implicit none", "    private", "    public :: run_group",
                "contains", "    subroutine run_group(name, matched)",
                "        character(len=*), intent(in) :: name",
                "        logical, intent(out) :: matched", "",
                "        matched = .true.", "        select case (name)"])
    for path in paths:
        out.extend([f'        case ("{path.stem}")',
                    f"            call {sym(path.stem)}()"])
    out.extend(["        case default", "            matched = .false.",
                "        end select", "    end subroutine run_group",
                f"end module {module}", ""])
    return "\n".join(out)


def registry(paths: list[pathlib.Path]) -> str:
    count = (len(paths) + GROUP_SIZE - 1) // GROUP_SIZE
    out = ["program test_ffc_suite",
           "    ! One case per process preserves independent STOP and exit status."]
    for index in range(count):
        out.extend([f"    use ffc_suite_group_{index:02d}, only: &",
                    f"        run_group_{index:02d} => run_group"])
    out.extend(["    implicit none", "    character(len=256) :: name",
                "    logical :: matched", "",
                "    if (command_argument_count() /= 1) then",
                '        print *, "usage: test_ffc_suite <test_name>"',
                "        stop 2", "    end if",
                "    call get_command_argument(1, name)", ""])
    for index in range(count):
        out.extend([f"    call run_group_{index:02d}(trim(name), matched)",
                    "    if (matched) goto 100"])
    out.extend(['    print *, "ffc_suite: no such case: "//trim(name)',
                "    stop 3", "100 continue", "end program test_ffc_suite", ""])
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("tests", nargs="*")
    args = parser.parse_args()
    targets = sorted(pathlib.Path(name) for name in args.tests) if args.tests else sorted(
        path for path in pathlib.Path("test").glob("test_*.f90") if path != MAIN)
    if not targets:
        parser.error("no test sources selected")
    try:
        changes = [(path, convert(path)) for path in targets]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    already = [path for path in pathlib.Path("test").glob("test_*.f90")
               if path != MAIN and converted_module(path)]
    registered = sorted(set(targets + already))
    print(f"{len(targets)} selected; {len(registered)} registered cases")
    if not args.apply:
        return 0
    for path, text in changes:
        if path.read_text() != text:
            path.write_text(text)
    # Bounded registry fanout respects fo's source/DAG dependency limits and
    # keeps each generated procedure below the project's 100-line cap.
    GROUPS.mkdir(parents=True, exist_ok=True)
    groups = {}
    for start in range(0, len(registered), GROUP_SIZE):
        index = start // GROUP_SIZE
        path = GROUPS / f"ffc_suite_group_{index:02d}.f90"
        groups[path] = group_source(index, registered[start:start + GROUP_SIZE])
    for stale in GROUPS.glob("ffc_suite_group_*.f90"):
        if stale not in groups:
            stale.unlink()
    for path, source in groups.items():
        if not path.exists() or path.read_text() != source:
            path.write_text(source)
    source = registry(registered)
    if not MAIN.exists() or MAIN.read_text() != source:
        MAIN.write_text(source)
    print(f"wrote {MAIN}; test bodies remain in their original source files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
