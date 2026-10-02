#!/usr/bin/env python3
"""Behavioral oracle for converting independent programs into dispatch cases."""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = """program original_program_name
    implicit none
    integer :: host_value
    character(len=16) :: host_text
    host_value = 4
    host_text = 'lexical scope'
    call run_case()
    if (host_value /= 9) stop 21
    if (trim(host_text) /= 'host preserved') stop 22
    print *, 'WITNESS: host association preserved'
contains
    subroutine run_case()
        host_value = host_value + 5
        host_text = 'host preserved'
    end subroutine run_case
end program original_program_name
"""
FAILURE = """program failed_case
    implicit none
    print *, 'WITNESS: explicit failure'
    stop 7
end program failed_case
"""


def invoke(fo: str, project: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, FO_JOBS="4", FO_TEST_REPORT_NAMES="1",
               TMPDIR=str(project))
    return subprocess.run([fo, *args], cwd=project, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=120)


def verdicts(output: str) -> dict[str, str]:
    return dict(re.findall(r"^TEST_RESULT\s+(\S+)\s+(PASS|FAIL|SKIP)\b",
                           output, re.MULTILINE))


def require(condition: bool, message: str, output: str = "") -> None:
    if not condition:
        raise AssertionError(message + "\n" + "\n".join(output.splitlines()[-25:]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fo", default="fo")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ffc-suite-oracle-", dir="/var/tmp") as name:
        project = pathlib.Path(name)
        (project / "test").mkdir()
        manifest = project / "fpm.toml"
        manifest.write_text('name="suite_oracle"\nversion="0.1.0"\n'
                            '[fortran]\nimplicit-typing=false\n')
        long_name = "test_host_" + "association_" * 5
        (project / "test" / (long_name + ".f90")).write_text(SOURCE)
        (project / "test/test_explicit_failure.f90").write_text(FAILURE)
        expected = {long_name: "PASS", "test_explicit_failure": "FAIL"}
        # Cross the scanner's 64-dependency limit: every case must retain its
        # separate verdict after the generator builds bounded registry groups.
        for index in range(65):
            case = f"test_independent_{index:02d}"
            source = SOURCE.replace("original_program_name", f"program_independent_{index}")
            source = source.replace("WITNESS: host", f"WITNESS: case {index} host")
            (project / "test" / (case + ".f90")).write_text(source)
            expected[case] = "PASS"
        baseline = invoke(args.fo, project, "test", "--all")
        require(verdicts(baseline.stdout) == expected,
                "standalone control must pass its host check and fail with STOP", baseline.stdout)
        conversion = subprocess.run(
            ["python3", str(ROOT / "tools/make_suite_cases.py"), "--apply"],
            cwd=project, text=True, capture_output=True, timeout=30)
        require(conversion.returncode == 0, "conversion failed", conversion.stderr)
        manifest.write_text(manifest.read_text() +
                            '[extra.fo]\ndispatcher="test_ffc_suite"\n')
        consolidated = invoke(args.fo, project, "test", "--all")
        require(verdicts(consolidated.stdout) == expected,
                "consolidation changed an independent program's verdict", consolidated.stdout)
        require("FAIL       7" in consolidated.stdout or "FAIL 7" in consolidated.stdout,
                "STOP 7 must remain a failing case with its exit status", consolidated.stdout)
        named = invoke(args.fo, project, "test", long_name)
        require(named.returncode == 0 and verdicts(named.stdout) == {long_name: "PASS"},
                "long filename must remain individually runnable", named.stdout)
        changed = project / "test" / (long_name + ".f90")
        changed.write_text(changed.read_text().replace("host_value /= 9", "host_value /= 99"))
        falsified = invoke(args.fo, project, "test", long_name)
        require(verdicts(falsified.stdout) == {long_name: "FAIL"},
                "editing the original source must affect the dispatched behavior", falsified.stdout)
        changed.write_text(changed.read_text().replace("host_value /= 99", "host_value /= 9"))
        restored = invoke(args.fo, project, "test", long_name)
        require(restored.returncode == 0 and verdicts(restored.stdout) == {long_name: "PASS"},
                "restoring the behavioral oracle must restore the passing verdict", restored.stdout)
        print("PASS: 67 independent statuses, host association, long names, original-source edits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
