from __future__ import annotations

import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from harnessgym.artifacts import read_json
from harnessgym.config import RunConfig
from harnessgym.orchestrator import Orchestrator
from harnessgym.postprocess import post_command_passed, run_post_command
from harnessgym.runners.fake_runner import FakeRunner


class PostCommandPassedTests(unittest.TestCase):
    def test_post_command_passed_returns_false_for_missing_json(self) -> None:
        self.assertFalse(post_command_passed({"status": "completed", "json": None}))

    def test_post_command_passed_returns_false_for_unparseable_json(self) -> None:
        self.assertFalse(post_command_passed({"status": "completed", "json": None, "return_code": 0}))

    def test_post_command_passed_returns_false_for_non_dict_json(self) -> None:
        self.assertFalse(post_command_passed({"status": "completed", "json": [1, 2, 3]}))

    def test_post_command_passed_returns_true_for_passed_status(self) -> None:
        self.assertTrue(post_command_passed({"status": "completed", "json": {"status": "passed"}}))

    def test_post_command_passed_returns_true_for_solved_status(self) -> None:
        self.assertTrue(post_command_passed({"status": "completed", "json": {"status": "solved"}}))

    def test_post_command_passed_returns_false_for_failed_status(self) -> None:
        self.assertFalse(post_command_passed({"status": "completed", "json": {"status": "failed"}}))

    def test_post_command_passed_returns_false_for_non_completed_status(self) -> None:
        self.assertFalse(post_command_passed({"status": "failed", "json": {"status": "passed"}}))

    def test_post_command_passed_returns_true_for_dict_without_status(self) -> None:
        self.assertTrue(post_command_passed({"status": "completed", "json": {"score": 1}}))


class PostAttemptNonJsonOutputTests(unittest.TestCase):
    def test_non_json_post_attempt_output_does_not_mark_solved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            verifier = workspace / "post_verify.py"
            verifier.write_text(
                'print("all good")\n',
                encoding="utf-8",
            )
            config = RunConfig.from_values(
                workspace=workspace,
                task_text="Solve with post verifier.",
                iterations=1,
                runner="fake",
                run_id="post-non-json",
                post_attempt_command=f"{shlex.quote(sys.executable)} post_verify.py",
                post_attempt_timeout="5s",
            )

            results = Orchestrator(FakeRunner()).run(config)

            self.assertEqual(len(results), 1)
            self.assertNotEqual(results[0].status, "solved")
            self.assertFalse(results[0].solved)
            result_data = read_json(Path(results[0].result_path))
            self.assertNotEqual(result_data.get("verification", {}).get("status"), "passed")


if __name__ == "__main__":
    unittest.main()
