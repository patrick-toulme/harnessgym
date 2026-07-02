from __future__ import annotations

import unittest

from harnessgym.artifacts import deep_merge


class DeepMergeTests(unittest.TestCase):
    def test_dicts_merge_recursively(self) -> None:
        result = deep_merge({"a": {"b": 1}}, {"a": {"c": 2}})
        self.assertEqual(result, {"a": {"b": 1, "c": 2}})

    def test_non_array_keys_replace(self) -> None:
        result = deep_merge({"tags": ["a", "b"]}, {"tags": ["c"]})
        self.assertEqual(result, {"tags": ["c"]})

    def test_used_harness_tools_unions(self) -> None:
        result = deep_merge({"used_harness_tools": ["probe", "inspect"]}, {"used_harness_tools": ["other_tool"]})
        self.assertEqual(result, {"used_harness_tools": ["inspect", "other_tool", "probe"]})

    def test_used_harness_artifacts_unions(self) -> None:
        result = deep_merge({"used_harness_artifacts": ["a.py", "b.py"]}, {"used_harness_artifacts": ["c.py"]})
        self.assertEqual(result, {"used_harness_artifacts": ["a.py", "b.py", "c.py"]})

    def test_union_deduplicates(self) -> None:
        result = deep_merge({"used_harness_tools": ["probe", "inspect"]}, {"used_harness_tools": ["probe", "other"]})
        self.assertEqual(result, {"used_harness_tools": ["inspect", "other", "probe"]})

    def test_union_when_base_is_empty(self) -> None:
        result = deep_merge({"used_harness_tools": []}, {"used_harness_tools": ["probe"]})
        self.assertEqual(result, {"used_harness_tools": ["probe"]})

    def test_union_when_update_is_not_list_replaces(self) -> None:
        result = deep_merge({"used_harness_tools": ["probe"]}, {"used_harness_tools": "all"})
        self.assertEqual(result, {"used_harness_tools": "all"})


if __name__ == "__main__":
    unittest.main()
