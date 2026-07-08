# GitHub Upload Instructions for QScout CCF-A Artifact

日期：2026-07-08

## Why GitHub Still Looks Unchanged

The current GitHub repository at `https://github.com/caizhongqi/qscout` has not received the local artifact/reproducibility fixes yet. The local workspace contains the new files, but this machine does not have `git` or GitHub CLI on `PATH`, so the changes cannot be pushed from here.

## Files That Must Be Added to GitHub

Minimum required upload set:

```text
requirements.txt
requirements-qpu.txt
pyproject.toml
.github/workflows/artifact-ci.yml
scripts/verify_ccfa_artifacts.py
scripts/build_github_release_artifact.py
tests/test_lightweight_artifacts.py
paper_artifacts/ccfa_20260707/**
QScout2_CCF_A_Review_Response_20260707.md
QScout2_CCFA_Strengthening_Update_20260707.md
```

Also update these existing files:

```text
README.md
.gitignore
qlea/__init__.py
qlea/code_completion_attack/benchmark.py
qlea/code_completion_attack/targets.py
run_llm_topconf_streaming_matrix.py
run_qbw_signal_correlation.py
```

## Local Verification Before Upload

Run:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_ccfa_artifacts.py
& "D:\ProgramData\py2\python.exe" run_theory_sanity.py
& "D:\ProgramData\py2\python.exe" -m unittest tests.test_lightweight_artifacts -v
```

Expected:

```text
CCF-A artifact verification passed.
unittest: OK
```

## Release ZIP

Build:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\build_github_release_artifact.py
```

Generated file:

```text
release_artifacts/qscout_ccfa_artifact_20260708.zip
```

Upload this ZIP as a GitHub Release asset if raw completion traces are too large for the repository.

## Git Commands on a Machine With Git Installed

```bash
git add README.md .gitignore requirements.txt requirements-qpu.txt pyproject.toml
git add .github/workflows/artifact-ci.yml
git add scripts/verify_ccfa_artifacts.py scripts/build_github_release_artifact.py
git add tests/test_lightweight_artifacts.py
git add paper_artifacts/ccfa_20260707
git add QScout2_CCF_A_Review_Response_20260707.md QScout2_CCFA_Strengthening_Update_20260707.md
git add qlea/__init__.py qlea/code_completion_attack/benchmark.py qlea/code_completion_attack/targets.py
git add run_llm_topconf_streaming_matrix.py run_qbw_signal_correlation.py
git commit -m "Add CCF-A reproducibility artifacts and objective-aligned QScout evidence"
git push origin main
```

After pushing, the GitHub repository should visibly contain dependency files, paper artifacts, tests, and CI. That is the minimum condition before asking anyone to reassess CCF-A readiness from the public repository.
