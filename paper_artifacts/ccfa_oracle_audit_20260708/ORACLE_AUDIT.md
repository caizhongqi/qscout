# QScout CCF-A Oracle Boundary Audit

This audit checks the committed detector/task-outcome ledger.  It does not claim
human-level exploit validation and does not commit generated-code payloads.

- Task outcome rows audited: 58230
- Boundary queue rows retained: 500

## Dataset Summary

| Artifact | Dataset | Task rows | Effective rate | Vulnerable nonfunctional | Functional not vulnerable | Mean attempts |
|---|---|---:|---:|---:|---:|---:|
| cyberseceval | cyberseceval | 7200 | 0.1233 | 0.0082 | 0.6317 | 5.57 |
| main | llmseceval | 29250 | 0.7204 | 0.0000 | 0.2670 | 4.11 |
| main | securityeval | 21780 | 0.8108 | 0.0000 | 0.1865 | 3.27 |

## Interpretation

- `effective_rate` is the committed Unsafe-and-Functional task outcome rate at the task-at-budget level.
- `vulnerable_nonfunctional_rate` estimates how often the vulnerability detector fires on code that the functionality heuristic does not accept.
- `functional_not_vulnerable_rate` estimates benign or safe functional completions.
- `oracle_boundary_queue.csv` is a hash-only queue for human/static/unit-test follow-up; it is not removed from the main result.
