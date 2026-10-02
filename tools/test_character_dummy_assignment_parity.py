#!/usr/bin/env python3
"""Byte-exact gfortran oracle for borrowed character dummy assignment (#348)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile


CASES = {
    "fixed": """program p
  implicit none
  character(len=8) :: s
  s = 'abcdefgh'
  call change(s)
  print *, s
  call change(s)
  print *, s
contains
  subroutine change(x)
    character(len=3), intent(inout) :: x
    x = 'XY'
    print *, x, len(x)
    x = x(2:3)//'12345'
    print '(a)', x
  end subroutine
end program
""",
    "assumed": """program p
  implicit none
  character(len=6) :: s
  s = 'abcdef'
  call change(s)
  print *, s
contains
  subroutine change(x)
    character(len=*), intent(inout) :: x
    x = x(2:)
    print *, x, len(x)
    call forward(x)
    print *, x
  end subroutine
  subroutine forward(y)
    character(len=*), intent(out) :: y
    y = 'XY'
  end subroutine
end program
""",
    "optional_fixed": """program p
  implicit none
  character(len=8) :: s
  s = 'abcdefgh'
  call change()
  call change(s)
  print *, s
contains
  subroutine change(x)
    character(len=3), optional, intent(inout) :: x
    if (present(x)) then
      print *, x, len(x)
      x = 'Z'
      print *, x, len(x)
    end if
  end subroutine
end program
""",
    "deferred_actual": """program p
  implicit none
  character(len=:), allocatable :: s
  s = 'abcdef'
  call change(s)
  print *, s, len(s)
  s = 'abcdefghijkl'
  call change(s)
  print *, s, len(s)
  deallocate(s)
contains
  subroutine change(x)
    character(len=*), intent(out) :: x
    x = 'XY'
    print *, x, len(x)
  end subroutine
end program
""",
}


def checked(command: list[str], cwd: Path | None = None) -> bytes:
    result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{command!r}: rc={result.returncode}\n"
                           f"{result.stderr.decode(errors='replace')}")
    return result.stdout


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = os.environ.get("CHAR_DUMMY_WORK")
    work = Path(destination or tempfile.mkdtemp(
        prefix="ffc-char-dummy-", dir="/var/tmp"))
    work.mkdir(parents=True, exist_ok=True)
    runs = int(os.environ.get("RUNS", "20"))
    if runs < 20:
        raise ValueError("RUNS must be at least 20")
    report = []
    for name, source in CASES.items():
        path = work / f"{name}.f90"
        path.write_text(source)
        reference, compiled = work / f"{name}.ref", work / f"{name}.ffc"
        checked(["gfortran", "-std=f2018", str(path), "-o", str(reference)])
        expected = checked([str(reference)])
        checked(["fo", "exec", "--no-build", "ffc", str(path), "-o",
                 str(compiled)], cwd=root)
        for run in range(runs):
            actual = checked([str(compiled)])
            if actual != expected:
                raise AssertionError(f"{name} run {run + 1}: "
                                     f"got={actual!r} expected={expected!r}")
        digest = hashlib.md5(compiled.read_bytes()).hexdigest()
        output_digest = hashlib.md5(expected).hexdigest()
        report.append(f"{name}\t{runs}\t{digest}\t{output_digest}")
        print(f"PASS {name}: {runs} runs; binary md5={digest}; "
              f"output md5={output_digest}")
    (work / "report.tsv").write_text("\n".join(report) + "\n")
    forbidden = work / "intent_in.f90"
    forbidden.write_text("""program p
  call change('abc')
contains
  subroutine change(x)
    character(len=*), intent(in) :: x
    x = 'XY'
  end subroutine
end program
""")
    for command in (["gfortran", "-std=f2018", str(forbidden), "-o",
                     str(work / "intent_in.ref")],
                    ["fo", "exec", "--no-build", "ffc", str(forbidden), "-o",
                     str(work / "intent_in.ffc")]):
        result = subprocess.run(command, cwd=root, capture_output=True, timeout=90)
        if result.returncode == 0:
            raise AssertionError(f"INTENT(IN) assignment accepted: {command!r}")
    print("PASS: INTENT(IN) assignment rejected by both compilers")
    print(f"PASS: {len(CASES) * runs} byte-exact runs; evidence {work}")


if __name__ == "__main__":
    main()
