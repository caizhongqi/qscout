# QScout CCF-A Compact Trace Artifact

This artifact is the committed raw-trace verification layer for the
CCF-A evidence package.  It does not commit full generated code or
model caches; instead it commits detector outcomes and hashes that
allow table reaggregation and local raw-output integrity checks.

- Main task-outcome rows: 51030
- CyberSecEval task-outcome rows: 7200
- Source file hash rows: 106
- Completion cache hash rows: 10789

Run:

```powershell
& "D:\ProgramData\py2\python.exe" scripts\verify_trace_artifact.py
```
