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


if __name__ == "__main__":
    unittest.main()
