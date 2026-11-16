"""Guards for nightly triage, with fake GitHub calls and temporary git history."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nightly_triage as nt

JOBS = list(nt.JOBS)
FIRST = JOBS[0]


class TriageTests(unittest.TestCase):
    def test_job_selection(self):
        self.assertEqual(nt.parse_jobs(""), JOBS)
        self.assertEqual(nt.parse_jobs(f" {FIRST},{FIRST}"), [FIRST])
        with self.assertRaises(ValueError):
            nt.parse_jobs("no-such-job")
        self.assertEqual(nt.title_jobs("nightly: all jobs"), set(JOBS))
        self.assertEqual(nt.title_jobs(f"nightly: jobs {FIRST}"), {FIRST})
        self.assertIsNone(nt.title_jobs("CI"))
        self.assertIsNone(nt.title_jobs("nightly: jobs no-such-job"))

    def test_failing_jobs_from_run(self):
        jobs = [{"name": f"Nightly {FIRST}", "conclusion": "failure"},
                {"name": "Nightly no-such-job", "conclusion": "failure"},
                {"name": "Plan", "conclusion": "failure"},
                {"name": "Report the failure", "conclusion": "failure"}]
        self.assertEqual(nt.failing_jobs(jobs), [FIRST])
        jobs[0]["conclusion"] = "success"
        self.assertEqual(nt.failing_jobs(jobs), [])

    def test_last_green_must_cover_the_job(self):
        runs = [
            {"databaseId": 9, "conclusion": "success", "headSha": "self",
             "displayTitle": "nightly: all jobs"},
            {"databaseId": 8, "conclusion": "success", "headSha": "other",
             "displayTitle": "nightly: jobs no-such-job"},
            {"databaseId": 7, "conclusion": "failure", "headSha": "red",
             "displayTitle": "nightly: all jobs"},
            {"databaseId": 6, "conclusion": "success", "headSha": "green",
             "displayTitle": f"nightly: jobs {FIRST}"}]
        self.assertEqual(nt.last_green(runs, FIRST, 9)["headSha"], "green")
        self.assertIsNone(nt.last_green(runs[:3], FIRST, 9))

    def test_report_and_closing(self):
        results = {FIRST: {"job": FIRST, "bad": "b" * 40, "good": "a" * 40,
                           "commits": 3, "pull": 42, "note": None,
                           "culprit": {"sha": "c" * 40, "subject": "Break it"}}}
        body = nt.failure_report(
            branch="main", bad="b" * 40, jobs=[FIRST, "setup-only"],
            job_urls={FIRST: "https://x/j"}, results=results,
            run_url="https://x/run", server="https://github.com", repo="o/r")
        self.assertIn(f"[{FIRST}](https://x/j): commit range", body)
        self.assertIn("first bad commit " + "c" * 40 + " Break it from #42", body)
        self.assertIn("compare/" + "a" * 40 + "..." + "b" * 40, body)
        self.assertIn("setup-only: failed before its command ran", body)
        self.assertNotIn("@", body)
        recorded = nt.issue_jobs([body, "a comment"])
        self.assertEqual(recorded, {FIRST, "setup-only"})
        self.assertFalse(nt.should_close([FIRST], recorded))
        self.assertTrue(nt.should_close([FIRST, "setup-only"], recorded))
        self.assertFalse(nt.should_close(JOBS, set()))

    def test_unknown_range_is_stated(self):
        results = {FIRST: {"job": FIRST, "bad": "b" * 40, "good": None,
                           "commits": 0, "culprit": None, "pull": None, "note": None}}
        body = nt.failure_report(
            branch="main", bad="b" * 40, jobs=[FIRST], job_urls={}, results=results,
            run_url="https://x/run", server="https://github.com", repo="o/r")
        self.assertIn("commit range is unknown", body)


class QueueTriageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environ = patch.dict(os.environ, {
            "GITHUB_EVENT_NAME": "merge_group",
            "GITHUB_REF_NAME": "gh-readonly-queue/main/group-a",
            "GITHUB_SHA": "b" * 40,
            "GITHUB_REPOSITORY": "example/ffc",
            "GITHUB_RUN_ID": "10",
            "GITHUB_SERVER_URL": "https://github.com",
            "TRIAGE_BASE_REF": "refs/heads/main",
            "TRIAGE_BASE_SHA": "a" * 40,
            "RUNNER_TEMP": str(self.root / "runner"),
        })
        self.environ.start()
        self.addCleanup(self.environ.stop)
        (self.root / "runner").mkdir()

    def test_queue_groups_share_the_base_branch_issue(self):
        issues, closed = [], []

        def github(*args, **kwargs):
            if args[:2] == ("api", "--paginate"):
                return json.dumps({"name": f"Nightly {FIRST}",
                                   "conclusion": "failure", "html_url": "https://x/job"})
            if args[:2] == ("issue", "list"):
                return [{"number": i + 1, "title": issue["title"]}
                        for i, issue in enumerate(issues)]
            if args[:2] == ("label", "create"):
                return ""
            if args[:2] == ("issue", "create"):
                issues.append({"title": args[args.index("--title") + 1],
                               "body": args[args.index("--body") + 1], "comments": []})
                return ""
            number = int(args[2])
            if args[:2] == ("issue", "comment"):
                issues[number - 1]["comments"].append({"body": args[-1]})
                return ""
            if args[:2] == ("issue", "view"):
                return issues[number - 1]
            if args[:2] == ("issue", "close"):
                closed.append(number)
                return ""
            self.fail(f"unexpected GitHub call: {args}")

        with patch.object(nt, "gh", side_effect=github), contextlib.redirect_stdout(io.StringIO()):
            nt.cmd_report(SimpleNamespace(results=self.root))
            os.environ["GITHUB_REF_NAME"] = "gh-readonly-queue/main/group-b"
            nt.cmd_report(SimpleNamespace(results=self.root))
            nt.cmd_success(SimpleNamespace(jobs=""))
            os.environ["TRIAGE_BASE_REF"] = "refs/heads/release/v1"
            nt.cmd_report(SimpleNamespace(results=self.root))
        self.assertEqual([issue["title"] for issue in issues],
                         ["Nightly failure on main", "Nightly failure on release/v1"])
        self.assertEqual(len(issues[0]["comments"]), 1)
        self.assertEqual(closed, [1])

    def test_invalid_queue_metadata_has_no_github_side_effect(self):
        for ref, sha in [("", ""), ("refs/heads/main", ""),
                         ("", "a" * 40), ("main", "a" * 40),
                         ("refs/heads/../main", "a" * 40),
                         ("refs/heads/main", "not-a-sha")]:
            with self.subTest(ref=ref, sha=sha), patch.dict(os.environ, {
                    "TRIAGE_BASE_REF": ref, "TRIAGE_BASE_SHA": sha}), \
                    patch.object(nt, "gh") as github:
                for command, args in [(nt.cmd_report, SimpleNamespace(results=self.root)),
                                      (nt.cmd_success, SimpleNamespace(jobs="")),
                                      (nt.cmd_bisect, SimpleNamespace(job=FIRST,
                                                                    out=self.root / "result.json"))]:
                    with self.assertRaises(ValueError):
                        command(args)
                github.assert_not_called()

    def test_legacy_history_lookup_is_unchanged(self):
        with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "schedule",
                                     "GITHUB_REF_NAME": "main",
                                     "TRIAGE_BASE_REF": "", "TRIAGE_BASE_SHA": ""}), \
                patch.object(nt, "gh", return_value=[]) as github, \
                contextlib.redirect_stdout(io.StringIO()):
            nt.cmd_bisect(SimpleNamespace(job=FIRST, out=self.root / "legacy.json"))
        args = github.call_args.args
        self.assertEqual(args[:2], ("run", "list"))
        self.assertEqual(args[args.index("--branch") + 1], "main")

    def test_queue_base_is_retested_before_bisection(self):
        history = self.root / "history"
        history.mkdir()
        original = Path.cwd()
        os.chdir(history)
        self.addCleanup(os.chdir, original)

        def git(*args):
            return subprocess.run(["git", *args], check=True, capture_output=True,
                                  text=True).stdout.strip()

        git("init", "--quiet")
        git("config", "user.name", "Synthetic test")
        git("config", "user.email", "test@example.invalid")
        state = history / "state"
        state.write_text("0")
        git("add", "state")
        git("commit", "--quiet", "-m", "Passing base")
        good = git("rev-parse", "HEAD")
        state.write_text("1")
        git("commit", "--quiet", "-am", "First failure")
        bad = git("rev-parse", "HEAD")
        git("commit", "--quiet", "--allow-empty", "-m", "Still failing")
        head = git("rev-parse", "HEAD")
        git("checkout", "--quiet", "--detach", good)
        git("commit", "--quiet", "--allow-empty", "-m", "Other branch")
        unrelated = git("rev-parse", "HEAD")
        git("checkout", "--quiet", head)
        probe = self.root / "probe.py"
        probe.write_text("import os, pathlib, subprocess, sys\n"
                         "with open(os.environ['PROBE_LOG'], 'a') as out:\n"
                         "    out.write(subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True))\n"
                         "sys.exit(int(pathlib.Path('state').read_text()))\n")
        log = self.root / "probe.log"
        out = self.root / "result.json"
        with patch.dict(os.environ, {"GITHUB_SHA": head, "PROBE_LOG": str(log)}), \
                patch.object(nt, "__file__", str(probe)), \
                patch.object(nt, "gh", return_value=[]) as github, \
                contextlib.redirect_stdout(io.StringIO()):
            os.environ["TRIAGE_BASE_SHA"] = good
            nt.cmd_bisect(SimpleNamespace(job=FIRST, out=out))
            result = json.loads(out.read_text())
            self.assertEqual(result["good"], good)
            self.assertEqual(result["culprit"]["sha"], bad)
            self.assertEqual(log.read_text().splitlines()[0], good)
            self.assertFalse(any(call.args[:2] == ("run", "list")
                                 for call in github.call_args_list))
            os.environ["TRIAGE_BASE_SHA"] = bad
            log.write_text("")
            nt.cmd_bisect(SimpleNamespace(job=FIRST, out=out))
            result = json.loads(out.read_text())
            self.assertIsNone(result["good"])
            self.assertIsNone(result["culprit"])
            self.assertEqual(log.read_text().splitlines(), [bad])
            for base in (head, unrelated, "f" * 40):
                os.environ["TRIAGE_BASE_SHA"] = base
                log.write_text("")
                nt.cmd_bisect(SimpleNamespace(job=FIRST, out=out))
                result = json.loads(out.read_text())
                self.assertIsNone(result["good"])
                self.assertIsNone(result["culprit"])
                self.assertEqual(log.read_text(), "")


if __name__ == "__main__":
    unittest.main()
