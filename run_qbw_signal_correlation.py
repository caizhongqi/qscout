"""Correlate boundary-witness scores with observed hard-label query utility."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from qlea.code_completion_attack.benchmark import (
    CodeAttackConfig,
    _calibrated_qql_candidate_pool,
    _comment_quantum_signals,
    _load_tasks,
    _minmax,
    _minmax_columns,
    _qql_priority_objective_comments,
    _quantum_boundary_witness_scores,
    _query_utility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", required=True, help="Comma-separated detail CSV paths.")
    parser.add_argument("--dataset", default="securityeval", choices=["internal", "securityeval", "llmseceval"])
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strategy-filter", default="qbw_qql_comment")
    parser.add_argument("--output-dir", default="outputs/qbw_signal_correlation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        task.task_id: task
        for task in _load_tasks(CodeAttackConfig(dataset=args.dataset, dataset_path=args.dataset_path))
    }
    rows = list(_iter_rows(args.details.split(",")))
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if args.strategy_filter and args.strategy_filter not in row.get("strategy", ""):
            continue
        key = (
            row.get("seed", str(args.seed)),
            row.get("strategy", ""),
            row.get("budget", ""),
            row.get("task_index", ""),
            row.get("task_id", ""),
        )
        grouped[key].append(row)

    signal_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, int, str, int], tuple[list[str], np.ndarray, np.ndarray, np.ndarray]] = {}
    for (seed_text, strategy, budget_text, _task_index, task_id), group in grouped.items():
        task = tasks.get(task_id)
        if task is None:
            continue
        seed = int(float(seed_text or args.seed))
        budget = int(float(budget_text or 0))
        cache_key = (task_id, budget, strategy, seed)
        if cache_key not in cache:
            candidates = _calibrated_qql_candidate_pool(task, budget, seed, strategy)
            signals, prior, x = _comment_quantum_signals(task, candidates, seed)
            cache[cache_key] = (candidates, signals, prior, x)
        candidates, signals, prior, x = cache[cache_key]
        fallback = np.asarray([0.18, 0.14, 0.08, 0.12, 0.10, 0.08, 0.12, 0.12, 0.04, 0.02], dtype=float)
        weights = fallback / float(fallback.sum())
        calibrated_quantum = _minmax(_minmax_columns(signals) @ weights)
        detector = _minmax(signals[:, 7])
        lexical = _minmax(signals[:, 6])
        variance_penalty = _minmax(signals[:, 1])
        priority_list = list(_qql_priority_objective_comments(task, exact=True, objective=True))
        priority_comments = set(priority_list)
        priority_signal = np.asarray([1.0 if comment in priority_comments else 0.0 for comment in candidates], dtype=float)
        priority_rank = {
            comment: 1.0 - (rank / max(len(priority_list), 1))
            for rank, comment in enumerate(priority_list)
        }
        priority_rank_signal = np.asarray([priority_rank.get(comment, 0.0) for comment in candidates], dtype=float)
        queried: list[int] = []
        outcomes: list[tuple[bool, bool, bool]] = []
        for row in sorted(group, key=lambda item: int(float(item.get("query_index", item.get("attempt", 0)) or 0))):
            comment = row.get("comment", "")
            try:
                index = candidates.index(comment)
            except ValueError:
                continue
            qbw, classical = _quantum_boundary_witness_scores(signals, prior, x, queried, outcomes)
            guarded_qbw = _minmax((1.0 - _minmax(qbw)) * detector + 0.35 * _minmax(qbw) * priority_rank_signal)
            actual_qbw_acquisition = (
                0.46 * _minmax(guarded_qbw)
                + 0.20 * _minmax(prior)
                + 0.14 * calibrated_quantum
                + 0.10 * detector
                + 0.08 * lexical
                + 0.08 * priority_rank_signal
                + 0.05 * priority_signal
                - 0.06 * variance_penalty
            )
            if task_id.startswith("llmseceval_"):
                actual_qbw_acquisition = actual_qbw_acquisition + 0.08 * priority_signal + 0.10 * priority_rank_signal
            vulnerable = _as_bool(row.get("vulnerable", "0"))
            functional = _as_bool(row.get("functional", "0"))
            effective = _as_bool(row.get("effective_vulnerable", "0"))
            utility = _query_utility(vulnerable, functional, effective)
            signal_rows.append(
                {
                    "source": row.get("_source", ""),
                    "strategy": strategy,
                    "budget": budget,
                    "task_id": task_id,
                    "cwe": task.cwe,
                    "comment": comment,
                    "query_utility": utility,
                    "effective_vulnerable": int(effective),
                    "qbw_score": float(qbw[index]) if len(qbw) else 0.0,
                    "guarded_qbw": float(guarded_qbw[index]) if len(qbw) else 0.0,
                    "actual_qbw_acquisition": float(actual_qbw_acquisition[index]) if len(qbw) else 0.0,
                    "classical_witness": float(classical[index]) if len(classical) else 0.0,
                    "qbw_detector_guard": float((1.0 - _minmax(qbw)[index]) * _minmax(signals[:, 7])[index]) if len(qbw) else 0.0,
                    "qql_prior": float(prior[index]),
                    "measurement_probability": float(signals[index, 0]),
                    "measurement_variance": float(signals[index, 1]),
                    "detector_alignment": float(signals[index, 7]),
                }
            )
            queried.append(index)
            outcomes.append((vulnerable, functional, effective))

    rows_path = out_dir / "qbw_signal_rows.csv"
    _write_csv(rows_path, signal_rows)
    summary = _summary(signal_rows)
    summary_path = out_dir / "qbw_signal_correlation_summary.csv"
    _write_csv(summary_path, summary)
    md_path = out_dir / "qbw_signal_correlation_summary.md"
    _write_markdown(md_path, summary, len(signal_rows))
    print(f"signal_rows: {rows_path}")
    print(f"summary: {summary_path}")
    print(f"markdown: {md_path}")


def _iter_rows(paths: Iterable[str]) -> Iterable[dict[str, str]]:
    for item in paths:
        path = Path(item.strip())
        if not path:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                yield row


def _as_bool(value: str | int | float | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    utility = np.asarray([float(row["query_utility"]) for row in rows], dtype=float)
    out = []
    for name in (
        "qbw_score",
        "guarded_qbw",
        "actual_qbw_acquisition",
        "classical_witness",
        "qbw_detector_guard",
        "qql_prior",
        "measurement_probability",
        "measurement_variance",
        "detector_alignment",
    ):
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        success_mask = np.asarray([bool(row["effective_vulnerable"]) for row in rows], dtype=bool)
        out.append(
            {
                "signal": name,
                "n": len(rows),
                "pearson": _pearson(values, utility),
                "spearman": _pearson(_rank(values), _rank(utility)),
                "auc": _auc(values, success_mask),
                "precision_at_10": _precision_at_k(values, success_mask, 10),
                "precision_at_50": _precision_at_k(values, success_mask, 50),
                "precision_at_10pct": _precision_at_k(values, success_mask, max(1, int(round(0.10 * len(values))))),
                "base_success_rate": _base_rate(success_mask),
                "lift_at_10": _lift_at_k(values, success_mask, 10),
                "lift_at_50": _lift_at_k(values, success_mask, 50),
                "lift_at_10pct": _lift_at_k(values, success_mask, max(1, int(round(0.10 * len(values))))),
                "hit_at_10": _hit_at_k(values, success_mask, 10),
                "hit_at_10pct": _hit_at_k(values, success_mask, max(1, int(round(0.10 * len(values))))),
                "success_mean": float(values[success_mask].mean()) if np.any(success_mask) else 0.0,
                "failure_mean": float(values[~success_mask].mean()) if np.any(~success_mask) else 0.0,
                "success_failure_delta": (
                    float(values[success_mask].mean() - values[~success_mask].mean())
                    if np.any(success_mask) and np.any(~success_mask)
                    else 0.0
                ),
            }
        )
    qbw = np.asarray([float(row["qbw_score"]) for row in rows], dtype=float)
    prior = np.asarray([float(row["qql_prior"]) for row in rows], dtype=float)
    detector = np.asarray([float(row["detector_alignment"]) for row in rows], dtype=float)
    combined = _minmax(qbw) + 0.35 * _minmax(prior) + 0.20 * _minmax(detector)
    success_mask = np.asarray([bool(row["effective_vulnerable"]) for row in rows], dtype=bool)
    out.append(
        {
            "signal": "qbw_plus_prior_detector",
            "n": len(rows),
            "pearson": _pearson(combined, utility),
            "spearman": _pearson(_rank(combined), _rank(utility)),
            "auc": _auc(combined, success_mask),
            "precision_at_10": _precision_at_k(combined, success_mask, 10),
            "precision_at_50": _precision_at_k(combined, success_mask, 50),
            "precision_at_10pct": _precision_at_k(combined, success_mask, max(1, int(round(0.10 * len(rows))))),
            "base_success_rate": _base_rate(success_mask),
            "lift_at_10": _lift_at_k(combined, success_mask, 10),
            "lift_at_50": _lift_at_k(combined, success_mask, 50),
            "lift_at_10pct": _lift_at_k(combined, success_mask, max(1, int(round(0.10 * len(rows))))),
            "hit_at_10": _hit_at_k(combined, success_mask, 10),
            "hit_at_10pct": _hit_at_k(combined, success_mask, max(1, int(round(0.10 * len(rows))))),
            "success_mean": float(combined[success_mask].mean()) if np.any(success_mask) else 0.0,
            "failure_mean": float(combined[~success_mask].mean()) if np.any(~success_mask) else 0.0,
            "success_failure_delta": (
                float(combined[success_mask].mean() - combined[~success_mask].mean())
                if np.any(success_mask) and np.any(~success_mask)
                else 0.0
            ),
        }
    )
    return sorted(out, key=lambda row: abs(float(row["spearman"])), reverse=True)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    x = x[mask] - float(x[mask].mean())
    y = y[mask] - float(y[mask].mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _precision_at_k(values: np.ndarray, success_mask: np.ndarray, k: int) -> float:
    if len(values) == 0 or k <= 0:
        return 0.0
    k = min(k, len(values))
    order = np.argsort(values)[::-1][:k]
    return float(np.asarray(success_mask, dtype=bool)[order].mean())


def _base_rate(success_mask: np.ndarray) -> float:
    if len(success_mask) == 0:
        return 0.0
    return float(np.asarray(success_mask, dtype=bool).mean())


def _lift_at_k(values: np.ndarray, success_mask: np.ndarray, k: int) -> float:
    base = _base_rate(success_mask)
    if base <= 1e-12:
        return 0.0
    return _precision_at_k(values, success_mask, k) / base


def _hit_at_k(values: np.ndarray, success_mask: np.ndarray, k: int) -> float:
    if len(values) == 0 or k <= 0:
        return 0.0
    k = min(k, len(values))
    order = np.argsort(values)[::-1][:k]
    return float(np.any(np.asarray(success_mask, dtype=bool)[order]))


def _auc(values: np.ndarray, success_mask: np.ndarray) -> float:
    success_mask = np.asarray(success_mask, dtype=bool)
    positives = int(success_mask.sum())
    negatives = int((~success_mask).sum())
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = _rank(np.asarray(values, dtype=float)) + 1.0
    positive_rank_sum = float(ranks[success_mask].sum())
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]], n: int) -> None:
    lines = [
        f"# QBW signal-utility correlation (n={n})",
        "",
        "| Signal | Pearson | Spearman | AUC | Precision@10 | Hit@10 | Precision@10% | Hit@10% | Success mean | Failure mean | Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {signal} | {pearson:.3f} | {spearman:.3f} | {auc:.3f} | {precision_at_10:.3f} | {hit_at_10:.3f} | {precision_at_10pct:.3f} | {hit_at_10pct:.3f} | {success_mean:.3f} | {failure_mean:.3f} | {success_failure_delta:.3f} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
