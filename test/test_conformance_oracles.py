#!/usr/bin/env python3
"""Check interactive LF cases against independent standard-Fortran references."""

import hashlib
import json
import os
from pathlib import Path
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
        source.write_bytes(b"independent input\n")
        command = [
            "python3", str(ACTION), "--cwd", str(self.root), "--timeout", "5",
            "--output", str(output), "--metadata", str(metadata), "--stdin",
            str(source), "--", "python3", "-c",
            "import sys; print(sys.stdin.read(), end=''); sys.exit(7)",
        ]
        result = subprocess.run(command, capture_output=True, check=False)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(output.read_bytes(), b"independent input\n")
        self.assertEqual(metadata.read_text(), "7\texit\t0\n")
        source.unlink()
        result = subprocess.run(command, capture_output=True, check=False)
        self.assertEqual(result.returncode, 127)
        self.assertEqual(metadata.read_text(), "127\texec-error\t0\n")

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
