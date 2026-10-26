#!/usr/bin/env python3
"""Check interactive LF cases against independent standard-Fortran references."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
GAUNTLET = PROJECT / "scripts/conformance_gauntlet.sh"
ACTION = PROJECT / "scripts/conformance_action.py"


class ConformanceOracleTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(
            prefix="ffc-oracle-test-", dir=os.environ.get("TMPDIR", "/var/tmp")
        )
        self.root = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def test_action_reads_private_input_and_preserves_exit(self):
        source = self.root / "stdin"
        output = self.root / "output"
        metadata = self.root / "metadata"
        metrics = self.root / "metrics"
        source.write_bytes(b"independent input\n")
        command = [
            "python3", str(ACTION), "--cwd", str(self.root), "--timeout", "5",
            "--output", str(output), "--metadata", str(metadata), "--stdin",
            str(source), "--metrics", str(metrics), "--metric-label", "oracle",
            "--", "python3", "-c",
            "import sys; memory=bytearray(12*1024*1024); "
            "print(sys.stdin.read(), end=''); sys.exit(7)",
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(output.read_bytes(), b"independent input\n")
        self.assertEqual(metadata.read_text(), "7\texit\t0\n")
        label, elapsed, peak_rss = metrics.read_text().strip().split("\t")
        self.assertEqual(label, "oracle")
        self.assertGreater(float(elapsed), 0)
        self.assertGreater(int(peak_rss), 12 * 1024)
        self.assertLess(int(peak_rss), 256 * 1024)
        source.unlink()
        result = subprocess.run(command, capture_output=True, check=False)
        self.assertEqual(result.returncode, 127)
        self.assertEqual(metadata.read_text(), "127\texec-error\t0\n")
        self.assertEqual(len(metrics.read_text().splitlines()), 2)

    def test_discovery_selects_newest_own_executable_with_portable_find(self):
        project = self.root / "own checkout"
        candidates = [project / "build" / name / "ffc" for name in ("old", "new")]
        for index, candidate in enumerate(candidates):
            candidate.parent.mkdir(parents=True)
            candidate.write_text("#!/bin/sh\nprintf '%s\\n' fixture-" + str(index))
            candidate.chmod(0o755)
        ignored = project / "build" / "ffc"
        ignored.write_text("not executable\n")
        os.utime(ignored, (2000000000, 2000000000))
        tools = self.root / "tools"
        tools.mkdir()
        real_find = shutil.which("find")
        wrapper = tools / "find"
        wrapper.write_text(
            "#!/bin/sh\nfor arg do\ncase $arg in -printf|-executable) exit 2;; esac\ndone\n"
            "exec " + shlex.quote(real_find) + ' "$@"\n'
        )
        wrapper.chmod(0o755)
        environment = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ["PATH"])
        for run in range(20):
            expected = run % 2
            for index, candidate in enumerate(candidates):
                stamp = 1000000000 + run * 2 + (index == expected)
                os.utime(candidate, (stamp, stamp))
            result = subprocess.run(
                ["bash", "-c", 'PROJECT_DIR="$1"; source "$2"; find_ffc',
                 "discovery", str(project), str(PROJECT / "scripts/lib_conformance.sh")],
                env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            selected = Path(result.stdout.strip())
            self.assertEqual(selected, candidates[expected].resolve())
            execution = subprocess.run([str(selected)], capture_output=True, check=True)
            self.assertEqual(execution.stdout, f"fixture-{expected}\n".encode())

    def test_full_suite_listing_does_not_require_a_compiler(self):
        fortfront = self.root / "fortfront"
        for directory in ("f90", "lf"):
            root = fortfront / "examples" / directory
            root.mkdir(parents=True)
            (root / "nested").mkdir()
            (root / "nested/ignored.f90").write_text("nested\n")
            (root / "directory.lf").mkdir()
            (root / "ignored.txt").write_text("ignored\n")
        environment = dict(os.environ, FFC_FORTFRONT_DIR=str(fortfront), TMPDIR=str(self.root))
        for run in range(20):
            suite = "fortfront-lf" if run % 2 else "fortfront-f90"
            root = fortfront / "examples" / ("lf" if run % 2 else "f90")
            suffix = ".lf" if run % 3 == 0 and run % 2 else ".f90"
            (root / f"case {20 - run:02d}{suffix}").write_text("fixture\n")
            expected = sorted(
                entry.name for entry in root.iterdir()
                if entry.is_file() and entry.suffix in ({".f90", ".lf"} if run % 2 else {".f90"})
            )
            result = subprocess.run(
                ["bash", str(GAUNTLET), "--suite", suite, "--list-files",
                 "--ffc", str(self.root / "absent-compiler")],
                cwd=PROJECT, env=environment, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), expected)
        environment["FFC_FORTFRONT_DIR"] = str(self.root / "absent-corpus")
        missing = subprocess.run(
            ["bash", str(GAUNTLET), "--suite", "fortfront-f90", "--list-files"],
            cwd=PROJECT, env=environment, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("not found", missing.stderr)

    def test_bash_bootstrap_preserves_arguments(self):
        wrapper = self.root / "bootstrap.sh"
        wrapper.write_text(
            "source " + shlex.quote(str(PROJECT / "scripts/lib_shell.sh")) + "\n"
            'ffc_require_bash "$@" || exit 1\nprintf "%s\\n" "$BASH_VERSION" "$@"\n'
        )
        arguments = ["space in a path", 'literal $(exit 9) `exit 8`', ""]
        result = subprocess.run(
            ["/bin/bash", str(wrapper)] + arguments,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        version = tuple(int(value) for value in lines[0].split(".")[:2])
        self.assertGreaterEqual(version, (4, 3))
        self.assertEqual(lines[1:], arguments)

    def test_read_statement_has_an_independent_reference(self):
        report = self.root / "report.jsonl"
        environment = dict(os.environ, TMPDIR=str(self.root))
        command = [
            "bash", str(GAUNTLET), "--suite", "fortfront-lf", "--jobs", "1",
            "--file", "read_statement.lf", "--report", str(report),
        ]
        if os.environ.get("FFC_TEST_BIN"):
            command += ["--ffc", os.environ["FFC_TEST_BIN"]]
        result = subprocess.run(
            command, cwd=PROJECT, env=environment, capture_output=True, text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = [json.loads(line) for line in report.read_text().splitlines()]
        case = next(record for record in records if "file" in record)
        self.assertEqual(case["status"], "PASS")
        self.assertEqual(case["ref_compile_action"], "executed")
        self.assertEqual(case["ref_run_action"], "executed")
        self.assertEqual(case["ffc_output_sha256"], case["ref_output_sha256"])
        self.assertNotEqual(case["ffc_output_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(records[-1]["noref"], 0)

        cache = self.root / "reference-cache"
        command += ["--ref-cache", str(cache)]
        cached = subprocess.run(
            command, cwd=PROJECT, env=environment, capture_output=True, text=True,
            check=False,
        )
        self.assertEqual(cached.returncode, 0, cached.stdout + cached.stderr)
        outputs = list(cache.rglob("*.out"))
        self.assertEqual(len(outputs), 1)
        wrong_output = b"a stale reference result\n"
        outputs[0].write_bytes(wrong_output)
        ready = outputs[0].with_suffix(".ready")
        ready.write_text(
            "2\t0\t0\t" + hashlib.sha256(wrong_output).hexdigest() + "\n"
        )
        rebuilt = subprocess.run(
            command, cwd=PROJECT, env=environment, capture_output=True, text=True,
            check=False,
        )
        self.assertEqual(rebuilt.returncode, 0, rebuilt.stdout + rebuilt.stderr)
        records = [json.loads(line) for line in report.read_text().splitlines()]
        case = next(record for record in records if "file" in record)
        self.assertEqual(case["status"], "PASS")
        self.assertEqual(case["ref_compile_action"], "executed")
        self.assertEqual(case["ref_run_action"], "executed")
        self.assertEqual(case["ffc_output_sha256"], case["ref_output_sha256"])
        self.assertEqual(records[-1]["noref"], 0)


if __name__ == "__main__":
    unittest.main()
