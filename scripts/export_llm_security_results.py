"""Export paper artifacts into a reviewer-friendly LLM-security results layout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MAIN_ROOT = Path("paper_artifacts/ccfa_20260707")
CYBER_ROOT = Path("paper_artifacts/ccfa_20260708")
TRACE_ROOT = Path("paper_artifacts/ccfa_trace_20260708")
ORACLE_ROOT = Path("paper_artifacts/ccfa_oracle_audit_20260708")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/llm_security")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    trace_dir = out_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = _read_csv(MAIN_ROOT / "main_seed_rows.csv") + _read_csv(CYBER_ROOT / "ccfa_seed_rows.csv")
    budget_rows = _read_csv(MAIN_ROOT / "main_budget_summary.csv") + _read_csv(CYBER_ROOT / "ccfa_budget_summary.csv")
    strongest_rows = _read_csv(MAIN_ROOT / "main_strongest_baseline_comparison.csv") + _read_csv(
        CYBER_ROOT / "ccfa_strongest_baseline_comparison.csv"
    )
    aulc_rows = _read_csv(MAIN_ROOT / "main_aulc_summary.csv") + _read_csv(CYBER_ROOT / "ccfa_aulc_summary.csv")

    summary_rows = _summary_rows(seed_rows, aulc_rows)
    _write_csv(out_dir / "summary.csv", summary_rows)
    _write_csv(out_dir / "budget_summary.csv", budget_rows)
    _write_csv(out_dir / "strongest_baseline_comparison.csv", strongest_rows)
    _write_csv(out_dir / "aulc_summary.csv", aulc_rows)

    trace_counts = _export_traces(trace_dir)
    oracle_files = _copy_oracle(out_dir)
    manifest = {
        "summary_rows": len(summary_rows),
        "seed_rows": len(seed_rows),
        "budget_rows": len(budget_rows),
        "strongest_baseline_rows": len(strongest_rows),
        "trace_jsonl_files": trace_counts,
        "oracle_files": oracle_files,
        "note": "Mirrors committed CCF-A artifacts; generated-code payloads are not included.",
    }
    (out_dir / "RESULTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(_readme(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _summary_rows(seed_rows: list[dict[str, str]], aulc_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(row["dataset"], row["model"], row["strategy"], int(row["budget"]))].append(row)
    aulc_lookup = {
        (row["dataset"], row["model"], row["strategy"]): row
        for row in aulc_rows
    }
    out = []
    for (dataset, model, strategy, budget), items in sorted(grouped.items()):
        unsafe = [_float(row["unsafe_and_functional_at_q"]) for row in items]
        queries = [_float(row["queries"]) for row in items]
        queries_per_task = [_float(row["queries_per_task"]) for row in items]
        q_success = [_float(row["q_at_success"]) for row in items if _float(row["q_at_success"]) > 0]
        raw_asr = [_float(row["raw_asr"]) for row in items]
        functional = [_float(row["functional_rate"]) for row in items]
        aulc = aulc_lookup.get((dataset, model, strategy), {}).get("aulc", "")
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "strategy": strategy,
                "display_strategy": _display_strategy(strategy),
                "budget": budget,
                "seeds": ",".join(str(int(float(row["seed"]))) for row in sorted(items, key=lambda item: int(float(item["seed"])))),
                "seed_count": len(items),
                "ASR_at_budget": _mean(unsafe),
                "UnsafeFunctional_at_budget": _mean(unsafe),
                "RawVulnerable_at_budget": _mean(raw_asr),
                "FunctionalRate_at_budget": _mean(functional),
                "Q_at_success": _mean(q_success) if q_success else 0.0,
                "AvgQueries": _mean(queries),
                "AvgQueriesPerTask": _mean(queries_per_task),
                "AULC": aulc,
                "metric_note": "Budgets are the committed protocol budgets B=4/8/16 or B=4/8 for CyberSecEval, not B=10/20/40.",
            }
        )
    return out


def _export_traces(trace_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    rows = _read_csv(TRACE_ROOT / "main_task_outcomes.csv") + _read_csv(TRACE_ROOT / "cyberseceval_task_outcomes.csv")
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, items in sorted(by_dataset.items()):
        path = trace_dir / f"{dataset}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in items:
                payload = {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": int(float(row["seed"])),
                    "budget": int(float(row["budget"])),
                    "strategy": row["strategy"],
                    "query_id": int(float(row["task_index"])),
                    "task_id": row["task_id"],
                    "language": row.get("language", ""),
                    "cwe": row.get("cwe", ""),
                    "attempts": int(float(row["attempts"])),
                    "candidate_pool_size": int(float(row.get("candidate_pool_size", "0") or 0)),
                    "label": {
                        "vulnerable": bool(int(float(row["vulnerable"]))),
                        "functional": bool(int(float(row["functional"]))),
                        "unsafe_functional": bool(int(float(row["effective_vulnerable"]))),
                    },
                    "attempt_to_success": int(float(row.get("attempt_to_success", "0") or 0)),
                    "task_key_sha256": row.get("task_key_sha256", ""),
                    "source_file_sha256": row.get("source_file_sha256", ""),
                    "output_payload_policy": "generated code omitted; see completion_cache_manifest.csv hashes",
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        counts[path.name] = len(items)
    return counts


def _copy_oracle(out_dir: Path) -> list[str]:
    copied = []
    oracle_out = out_dir / "oracle_audit"
    oracle_out.mkdir(parents=True, exist_ok=True)
    for path in ORACLE_ROOT.glob("*"):
        if path.is_file():
            target = oracle_out / path.name
            target.write_bytes(path.read_bytes())
            copied.append(str(target.relative_to(out_dir)).replace("\\", "/"))
    return copied


def _readme(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "# LLM Security Results Mirror",
            "",
            "This directory mirrors the committed CCF-A artifacts in a security-paper-friendly layout.",
            "It is generated by `scripts/export_llm_security_results.py` and does not rerun models.",
            "",
            "## Files",
            "",
            "- `summary.csv`: seed-averaged ASR/UnsafeFunctional/Q@Success/AULC rows.",
            "- `budget_summary.csv`: method-by-budget summaries copied from the artifact tables.",
            "- `strongest_baseline_comparison.csv`: QScout-QBW vs the strongest non-quantum baseline.",
            "- `aulc_summary.csv`: area-under-low-budget-curve summaries.",
            "- `traces/*.jsonl`: task-level hard-label outcomes with hashes, no generated-code payloads.",
            "- `oracle_audit/`: detector boundary audit copied from `paper_artifacts`.",
            "",
            "## Scope",
            "",
            "The committed protocol budgets are B=4/8/16 for SecurityEval and LLMSecEval, and B=4/8 for CyberSecEval.",
            "The review-requested B=10/20/40 layout is not fabricated here; those budgets require a separate run.",
            "",
            "## Manifest",
            "",
            "```json",
            json.dumps(manifest, indent=2),
            "```",
            "",
        ]
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(value: str) -> float:
    return float(value) if value not in {"", None} else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _display_strategy(strategy: str) -> str:
    labels = {
        "fair_random_comment": "Random Search",
        "fair_risk_prior_comment": "Risk Prior",
        "classical_active_comment": "Classical Active",
        "insec_fixed_pool_comment": "INSEC-style Fixed-Pool Search",
        "aot_ensemble_fixed_pool_comment": "AOT-style Ensemble",
        "classical_boundary_witness_comment": "Classical Boundary Witness",
        "qscout_qbw_comment": "QScout-QBW",
    }
    return labels.get(strategy, strategy)


if __name__ == "__main__":
    main()
