"""Build a GitHub upload package for the QScout CCF-A artifact.

The package is intentionally compact: it contains source/code changes required
to reproduce the table-level evidence, committed CSV/Markdown artifacts, CI
configuration, and verification scripts. It excludes raw LLM completions and
model caches.

Run:
    python scripts/build_github_release_artifact.py
"""

from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "release_artifacts"
OUT_ZIP = OUT_DIR / "qscout_ccfa_artifact_20260708.zip"

FILES = [
    ".gitignore",
    ".github/workflows/artifact-ci.yml",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "requirements-qpu.txt",
    "pyproject.toml",
    "QScout2_CCF_A_Review_Response_20260707.md",
    "QScout2_CCFA_Strengthening_Update_20260707.md",
    "generate_ccfa_protocol_tables.py",
    "run_llm_topconf_streaming_matrix.py",
    "run_qbw_signal_correlation.py",
    "run_theory_sanity.py",
    "qlea/__init__.py",
    "qlea/theory.py",
    "qlea/quantum_boundary_witness.py",
    "qlea/code_completion_attack/benchmark.py",
    "qlea/code_completion_attack/targets.py",
    "qlea/code_completion_attack/detectors.py",
    "qlea/code_completion_attack/securityeval.py",
    "qlea/code_completion_attack/llmseceval.py",
    "qlea/code_completion_attack/tasks.py",
    "qlea/llm_safety/embeddings.py",
    "scripts/verify_ccfa_artifacts.py",
    "scripts/build_github_release_artifact.py",
    "tests/test_lightweight_artifacts.py",
]

ARTIFACT_DIRS = [
    "paper_artifacts/ccfa_20260707",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in FILES:
            _write_file(archive, rel)
        for rel_dir in ARTIFACT_DIRS:
            for path in sorted((ROOT / rel_dir).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT).as_posix())
    print(OUT_ZIP)


def _write_file(archive: zipfile.ZipFile, rel: str) -> None:
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"missing required release file: {rel}")
    archive.write(path, rel)


if __name__ == "__main__":
    main()
