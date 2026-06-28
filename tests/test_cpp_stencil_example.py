from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "cpp_stencil_kernel_task"


def compiler_available() -> bool:
    return any(shutil.which(candidate) for candidate in ("c++", "clang++", "g++"))


@unittest.skipUnless(compiler_available(), "C++ compiler is required for the stencil example")
class CppStencilExampleTests(unittest.TestCase):
    def test_dev_benchmark_and_trace_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_path = Path(temp_dir) / "trace.json"
            assembly_path = Path(temp_dir) / "kernel.s"
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmark.py",
                    "--json",
                    "--mode",
                    "dev",
                    "--trace",
                    str(trace_path),
                    "--assembly",
                    str(assembly_path),
                ],
                cwd=EXAMPLE,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "passed")
            self.assertGreater(payload["best_cycles"], 0)
            self.assertLessEqual(payload["max_abs"], 2.5e-5)
            self.assertTrue(trace_path.exists())
            self.assertTrue(assembly_path.exists())
            self.assertIn("instruction_lines", payload["assembly"])
            self.assertGreater(payload["assembly"]["instruction_lines"], 0)

    def test_final_verifier_passes(self) -> None:
        result = subprocess.run(
            [sys.executable, "verifier.py"],
            cwd=EXAMPLE,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "passed")
        self.assertGreater(payload["best_cycles"], 0)

    def test_benchmark_rejects_timer_tampering_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            shutil.copytree(EXAMPLE, workspace, ignore=shutil.ignore_patterns(".harnessgym_build"))
            (workspace / "kernel.cpp").write_text(
                '#include "kernel.h"\n'
                "extern \"C\" void stencil_step(const float*, float*, int, int, float) {\n"
                "  // PyRun_SimpleString would allow benchmark timer tampering.\n"
                "}\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "benchmark.py", "--json", "--mode", "dev"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )

            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["source_integrity"], "failed")
            self.assertIn("PyRun_", payload["errors"][0])

    def test_benchmark_rejects_returning_early_on_repeated_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            shutil.copytree(EXAMPLE, workspace, ignore=shutil.ignore_patterns(".harnessgym_build"))
            (workspace / "kernel.cpp").write_text(
                r'''
#include "kernel.h"

static int calls = 0;

extern "C" void stencil_step(const float* src, float* dst, int rows, int cols, float alpha) {
    if (calls++ > 0) {
        return;
    }
    const float side = alpha * 0.125f;
    const float center_weight = 1.0f - alpha * 0.5f;
    auto load = [&](int r, int c) {
        if (r < 0) r = 0;
        if (c < 0) c = 0;
        if (r >= rows) r = rows - 1;
        if (c >= cols) c = cols - 1;
        return src[r * cols + c];
    };
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float x = load(r, c);
            dst[r * cols + c] = center_weight * x + side * (
                load(r - 1, c) + load(r + 1, c) + load(r, c - 1) + load(r, c + 1)
            );
        }
    }
}
''',
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "benchmark.py", "--json", "--mode", "dev"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )

            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "failed")
            self.assertGreater(payload["cases"][0]["failures"][0]["max_abs"], 1.0)


if __name__ == "__main__":
    unittest.main()
