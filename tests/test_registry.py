import tempfile
import unittest
from pathlib import Path

from harnessgym.artifacts import ensure_harness_dirs, sync_registry_from_files
from harnessgym.models import Artifact
from harnessgym.registry import add_or_update_artifact, load_registry, save_registry


class RegistryTests(unittest.TestCase):
    def test_registry_roundtrip_and_filesystem_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            ensure_harness_dirs(tmp_path)
            registry = load_registry(tmp_path)
            add_or_update_artifact(
                registry,
                Artifact(
                    id="skill:.harnessgym/skills/demo/SKILL.md",
                    kind="skill",
                    path=".harnessgym/skills/demo/SKILL.md",
                    description="Demo skill",
                    iteration=1,
                ),
            )
            save_registry(tmp_path, registry)

            loaded = load_registry(tmp_path)
            self.assertEqual(loaded.artifacts[0].description, "Demo skill")

            tool = tmp_path / ".harnessgym" / "tools" / "probe.py"
            tool.write_text("print('probe')\n", encoding="utf-8")
            docs = tmp_path / ".harnessgym" / "docs" / "brief.md"
            docs.parent.mkdir(parents=True, exist_ok=True)
            docs.write_text("brief\n", encoding="utf-8")
            generated_test = tmp_path / ".harnessgym" / "tests" / "test_probe.py"
            generated_test.parent.mkdir(parents=True, exist_ok=True)
            generated_test.write_text("def test_probe():\n    assert True\n", encoding="utf-8")
            skill = tmp_path / ".harnessgym" / "skills" / "demo" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("---\nname: demo\ndescription: demo\n---\n", encoding="utf-8")
            extra_skill_file = skill.parent / "notes.md"
            extra_skill_file.write_text("notes\n", encoding="utf-8")
            mcp = tmp_path / ".harnessgym" / "mcp" / "server" / "mcp.json"
            mcp.parent.mkdir(parents=True)
            mcp.write_text('{"name":"server","command":"python3"}\n', encoding="utf-8")
            extra_mcp_file = mcp.parent / "server.py"
            extra_mcp_file.write_text("print('server')\n", encoding="utf-8")
            synced = sync_registry_from_files(tmp_path, loaded, iteration=2)

            self.assertIsNotNone(synced.get_artifact("tool:.harnessgym/tools/probe.py"))
            self.assertIsNotNone(synced.get_artifact("docs:.harnessgym/docs/brief.md"))
            self.assertIsNotNone(synced.get_artifact("test:.harnessgym/tests/test_probe.py"))
            self.assertIsNotNone(synced.get_artifact("skill:.harnessgym/skills/demo/SKILL.md"))
            self.assertIsNotNone(synced.get_artifact("mcp:.harnessgym/mcp/server/mcp.json"))
            self.assertIsNone(synced.get_artifact("skill:.harnessgym/skills/demo/notes.md"))
            self.assertIsNone(synced.get_artifact("mcp:.harnessgym/mcp/server/server.py"))
            self.assertIsNotNone(
                load_registry(tmp_path).get_artifact("skill:.harnessgym/skills/demo/SKILL.md")
            )

    def test_registry_loads_codex_style_artifact_without_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            registry_path = tmp_path / ".harnessgym" / "registry.json"
            registry_path.parent.mkdir(parents=True)
            registry_path.write_text(
                """{
  "version": 1,
  "updated_at": "2026-05-17T00:00:00+00:00",
  "metadata": {},
  "artifacts": [
    {
      "kind": "docs",
      "name": "brief",
      "path": ".harnessgym/docs/brief.md",
      "description": "Verifier brief"
    }
  ]
}
""",
                encoding="utf-8",
            )

            registry = load_registry(tmp_path)

            artifact = registry.get_artifact("docs:.harnessgym/docs/brief.md")
            self.assertIsNotNone(artifact)
            assert artifact is not None
            self.assertEqual(artifact.metadata["name"], "brief")
