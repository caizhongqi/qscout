from __future__ import annotations

import subprocess
import sys
import unittest


class LightweightArtifactTests(unittest.TestCase):
    def test_qlea_top_level_import_is_lightweight(self) -> None:
        code = (
            "import sys; "
            "import qlea; "
            "import qlea.theory; "
            "print(int('sklearn' in sys.modules), int('torch' in sys.modules))"
        )
        result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "0 0")

    def test_committed_ccfa_artifacts_verify(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/verify_ccfa_artifacts.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("CCF-A artifact verification passed.", result.stdout)

    def test_cyberseceval_subset_loader(self) -> None:
        code = (
            "from qlea.code_completion_attack.cyberseceval import build_cyberseceval_autocomplete_tasks; "
            "tasks=build_cyberseceval_autocomplete_tasks('paper_artifacts/ccfa_20260708/cyberseceval_autocomplete_subset_120.json'); "
            "print(len(tasks), len({t.language for t in tasks}), len({t.cwe for t in tasks}))"
        )
        result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
        count, languages, cwes = map(int, result.stdout.strip().split())
        self.assertEqual(count, 120)
        self.assertGreaterEqual(languages, 8)
        self.assertGreaterEqual(cwes, 40)


if __name__ == "__main__":
    unittest.main()
