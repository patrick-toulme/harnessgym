import tempfile
import unittest
from pathlib import Path

from harnessgym.config import RunConfig
from harnessgym.orchestrator import is_solved_and_verified, extract_score, reached_stop_score


class ScoreTests(unittest.TestCase):
    def test_extract_score_from_common_locations(self) -> None:
        self.assertEqual(extract_score({"score": "1.25"}, "best_ms"), 1.25)
        self.assertEqual(extract_score({"metrics": {"best_ms": 0.75}}, "best_ms"), 0.75)
        self.assertEqual(extract_score({"verification": {"score": 2}}, "score"), 2.0)
        self.assertIsNone(extract_score({"metrics": {"score": "nope"}}, "score"))

    def test_reached_stop_score_lower_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunConfig.from_values(
                workspace=Path(temp_dir),
                task_text="Optimize.",
                stop_score=1.0,
                score_key="best_ms",
            )

            self.assertTrue(reached_stop_score({"metrics": {"best_ms": 0.9}}, config))
            self.assertFalse(reached_stop_score({"metrics": {"best_ms": 1.1}}, config))

    def test_reached_stop_score_higher_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunConfig.from_values(
                workspace=Path(temp_dir),
                task_text="Optimize.",
                stop_score=10.0,
                score_key="throughput",
                higher_is_better=True,
            )

            self.assertTrue(reached_stop_score({"metrics": {"throughput": 12}}, config))
            self.assertFalse(reached_stop_score({"metrics": {"throughput": 8}}, config))

    def test_verified_result_can_still_have_unreached_optimization_score(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = RunConfig.from_values(
                workspace=Path(temp_dir),
                task_text="Optimize.",
                stop_score=6.0,
                score_key="best_ms",
            )
            result = {"status": "solved", "verified": True, "metrics": {"best_ms": 20.0}}

            self.assertTrue(is_solved_and_verified(result))
            self.assertFalse(reached_stop_score(result, config))
