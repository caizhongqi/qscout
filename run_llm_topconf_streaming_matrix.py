"""Streaming/resumable runner for top-conference code-LLM attack matrices.

Unlike the batch runner, this writes every query and every completed task to
disk immediately.  Long HF runs can therefore be interrupted and resumed
without losing completed task-level evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

from qlea.env import load_local_env
from qlea.code_completion_attack.benchmark import (
    _comment_quantum_signals,
    _is_early_feedback_qql,
    _is_global_feedback_qql,
    _is_online_calibrated_qql,
    _is_qbw_hard_rescue_task,
    _load_tasks,
    _preview,
    _prompt,
    _query_utility,
    _select_comments_for_task,
    _select_online_calibrated_qql_comment,
    _write_completion,
    CodeAttackConfig,
)
from qlea.code_completion_attack.detectors import is_vulnerable_completion, looks_functional
from qlea.code_completion_attack.targets import make_code_target
from qlea.code_completion_attack.tasks import ATTACK_COMMENTS


def main() -> None:
    load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="hf", choices=["offline", "hf", "openai", "gemini"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--dataset", default="securityeval", choices=["internal", "securityeval", "llmseceval"])
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--budgets", default="16")
    parser.add_argument("--seeds", default="7,19,31,43,59")
    parser.add_argument("--strategies", default="qfrontier_qsfa_comment,qbw_qql_comment,helstrom_qbw_qql_comment")
    parser.add_argument("--output-dir", default="outputs/llm_topconf_streaming_matrix_20260705")
    parser.add_argument("--prompt-mode", default="instruction", choices=["auto", "raw", "instruction"])
    parser.add_argument("--prompt-version", default="")
    parser.add_argument("--max-new-tokens", default="80")
    parser.add_argument("--hard-max-new-tokens", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    old_env = {name: os.environ.get(name) for name in ("HF_MODEL", "HF_PROMPT_MODE", "HF_MAX_NEW_TOKENS", "HF_CODE_PROMPT_VERSION")}
    try:
        if args.target == "hf":
            os.environ["HF_MODEL"] = args.model
            os.environ["HF_PROMPT_MODE"] = args.prompt_mode
            os.environ["HF_MAX_NEW_TOKENS"] = args.max_new_tokens
            if args.prompt_version:
                os.environ["HF_CODE_PROMPT_VERSION"] = args.prompt_version

        target = make_code_target(args.target)
        all_tasks = _load_tasks(
            CodeAttackConfig(
                target=args.target,
                dataset=args.dataset,
                dataset_path=args.dataset_path,
            )
        )
        task_items = list(enumerate(all_tasks))
        if args.task_start > 0:
            task_items = task_items[args.task_start :]
        if args.task_limit > 0:
            task_items = task_items[: args.task_limit]
        if args.max_tasks > 0:
            task_items = task_items[: args.max_tasks]
        summary_total_tasks = len(all_tasks) if args.task_start > 0 or args.task_limit > 0 else len(task_items)
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)

        for seed in _parse_ints(args.seeds):
            run_dir = root / f"{args.dataset}_{_slug(args.model if args.target == 'hf' else args.target)}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _run_one_seed(
                target=target,
                target_arg=args.target,
                model=args.model if args.target == "hf" else args.target,
                dataset=args.dataset,
                task_items=task_items,
                total_tasks=summary_total_tasks,
                seed=seed,
                budgets=_parse_ints(args.budgets),
                strategies=_parse_strings(args.strategies),
                run_dir=run_dir,
                rerun_completed=args.rerun_completed,
                base_max_new_tokens=args.max_new_tokens,
                hard_max_new_tokens=args.hard_max_new_tokens,
                batch_size=args.batch_size,
            )
        _merge_summaries(root)
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_one_seed(
    *,
    target,
    target_arg: str,
    model: str,
    dataset: str,
    task_items,
    total_tasks: int,
    seed: int,
    budgets: tuple[int, ...],
    strategies: tuple[str, ...],
    run_dir: Path,
    rerun_completed: bool,
    base_max_new_tokens: str,
    hard_max_new_tokens: str,
    batch_size: int,
) -> None:
    if batch_size > 1 and target_arg == "hf" and not hard_max_new_tokens:
        _run_one_seed_batched(
            target=target,
            target_arg=target_arg,
            model=model,
            dataset=dataset,
            task_items=task_items,
            total_tasks=total_tasks,
            seed=seed,
            budgets=budgets,
            strategies=strategies,
            run_dir=run_dir,
            rerun_completed=rerun_completed,
            batch_size=batch_size,
        )
        return
    completion_dir = run_dir / "completions"
    completion_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir.parent / "completion_cache.csv"
    completion_cache = _load_completion_cache(cache_path)
    detail_path = run_dir / "streaming_detail.csv"
    outcome_path = run_dir / "streaming_task_outcomes.csv"
    summary_path = run_dir / "streaming_summary.csv"
    completed = set()
    if outcome_path.exists() and not rerun_completed:
        for row in _read_csv(outcome_path):
            completed.add((row["budget"], row["strategy"], row.get("task_index", ""), row["task_id"]))

    for budget in budgets:
        for strategy in strategies:
            global_signal_rows: list[np.ndarray] = []
            global_signal_utilities: list[float] = []
            for task_index, task in task_items:
                task_key = (str(budget), strategy, str(task_index), task.task_id)
                if task_key in completed:
                    continue
                task_uid = f"{task_index:04d}_{task.task_id}"
                comments = _select_comments_for_task(strategy, budget, seed, task)
                candidate_pool_size = len(comments)
                best_vulnerable = False
                best_functional = False
                best_effective = False
                attempt_to_success = 0
                queried_indices: list[int] = []
                online_outcomes: list[tuple[bool, bool, bool]] = []
                attempts = 0
                for _ in range(min(budget, len(comments))):
                    if _is_online_calibrated_qql(strategy):
                        comment_index = _select_online_calibrated_qql_comment(
                            task,
                            comments,
                            queried_indices,
                            online_outcomes,
                            budget,
                            seed,
                            strategy=strategy,
                            early_feedback=_is_early_feedback_qql(strategy),
                            global_history_rows=global_signal_rows if _is_global_feedback_qql(strategy) else None,
                            global_history_utilities=global_signal_utilities if _is_global_feedback_qql(strategy) else None,
                        )
                        queried_indices.append(comment_index)
                        comment = comments[comment_index]
                    else:
                        comment_index = attempts
                        comment = comments[comment_index]
                    attempts += 1
                    prompt_text = _prompt(task, comment)
                    if target_arg == "hf" and hard_max_new_tokens:
                        os.environ["HF_MAX_NEW_TOKENS"] = hard_max_new_tokens if _is_qbw_hard_rescue_task(task) else base_max_new_tokens
                    cache_key = _completion_cache_key(
                        target_name=target.name,
                        model=model,
                        prompt_text=prompt_text,
                        max_new_tokens=os.environ.get("HF_MAX_NEW_TOKENS", ""),
                    )
                    completion = completion_cache.get(cache_key)
                    if completion is None:
                        completion = target.complete(task, prompt_text)
                        completion_cache[cache_key] = completion
                        _append_csv(
                            cache_path,
                            {
                                "cache_key": cache_key,
                                "dataset": dataset,
                                "model": model,
                                "target_name": target.name,
                                "max_new_tokens": os.environ.get("HF_MAX_NEW_TOKENS", ""),
                                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                                "completion": completion,
                            },
                        )
                    vulnerable = is_vulnerable_completion(task, completion)
                    functional = looks_functional(completion)
                    effective = vulnerable and functional
                    if _is_online_calibrated_qql(strategy):
                        online_outcomes.append((vulnerable, functional, effective))
                        if _is_global_feedback_qql(strategy):
                            signal_rows, _, _ = _comment_quantum_signals(task, comments, seed)
                            global_signal_rows.append(signal_rows[int(comment_index)])
                            global_signal_utilities.append(_query_utility(vulnerable, functional, effective))
                    completion_path = _write_completion(
                        completion_dir,
                        target_arg,
                        strategy,
                        budget,
                        task_uid,
                        attempts,
                        completion,
                    )
                    generated_path = _write_completion(
                        completion_dir,
                        target_arg,
                        strategy,
                        budget,
                        task_uid + "_full",
                        attempts,
                        prompt_text + completion,
                    )
                    _append_csv(
                        detail_path,
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "target_name": target.name,
                            "budget": budget,
                            "strategy": strategy,
                            "task_index": task_index,
                            "task_id": task.task_id,
                            "language": task.language,
                            "cwe": task.cwe,
                            "attempt": attempts,
                            "comment_id": ATTACK_COMMENTS.index(comment) if comment in ATTACK_COMMENTS else -1,
                            "comment_index": comment_index,
                            "candidate_pool_size": candidate_pool_size,
                            "comment": comment,
                            "vulnerable": int(vulnerable),
                            "functional": int(functional),
                            "effective_vulnerable": int(effective),
                            "completion_path": str(completion_path),
                            "generated_code_path": str(generated_path),
                            "completion_preview": _preview(completion),
                        },
                    )
                    best_vulnerable = best_vulnerable or vulnerable
                    best_functional = best_functional or functional
                    best_effective = best_effective or effective
                    if effective and attempt_to_success == 0:
                        attempt_to_success = attempts
                    if best_effective:
                        break
                _append_csv(
                    outcome_path,
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "budget": budget,
                        "strategy": strategy,
                        "task_index": task_index,
                        "task_id": task.task_id,
                        "language": task.language,
                        "cwe": task.cwe,
                        "attempts": attempts,
                        "candidate_pool_size": candidate_pool_size,
                        "vulnerable": int(best_vulnerable),
                        "functional": int(best_functional),
                        "effective_vulnerable": int(best_effective),
                        "attempt_to_success": attempt_to_success,
                    },
                )
                completed.add(task_key)
                _write_summary(outcome_path, summary_path, total_tasks=total_tasks)
    _write_summary(outcome_path, summary_path, total_tasks=total_tasks)


def _run_one_seed_batched(
    *,
    target,
    target_arg: str,
    model: str,
    dataset: str,
    task_items,
    total_tasks: int,
    seed: int,
    budgets: tuple[int, ...],
    strategies: tuple[str, ...],
    run_dir: Path,
    rerun_completed: bool,
    batch_size: int,
) -> None:
    completion_dir = run_dir / "completions"
    completion_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir.parent / "completion_cache.csv"
    completion_cache = _load_completion_cache(cache_path)
    detail_path = run_dir / "streaming_detail.csv"
    outcome_path = run_dir / "streaming_task_outcomes.csv"
    summary_path = run_dir / "streaming_summary.csv"
    completed = set()
    if outcome_path.exists() and not rerun_completed:
        for row in _read_csv(outcome_path):
            completed.add((row["budget"], row["strategy"], row.get("task_index", ""), row["task_id"]))

    for budget in budgets:
        for strategy in strategies:
            global_signal_rows: list[np.ndarray] = []
            global_signal_utilities: list[float] = []
            states = []
            for task_index, task in task_items:
                task_key = (str(budget), strategy, str(task_index), task.task_id)
                if task_key in completed:
                    continue
                comments = _select_comments_for_task(strategy, budget, seed, task)
                states.append(
                    {
                        "task_index": task_index,
                        "task": task,
                        "task_key": task_key,
                        "task_uid": f"{task_index:04d}_{task.task_id}",
                        "comments": comments,
                        "candidate_pool_size": len(comments),
                        "best_vulnerable": False,
                        "best_functional": False,
                        "best_effective": False,
                        "attempt_to_success": 0,
                        "queried_indices": [],
                        "online_outcomes": [],
                        "attempts": 0,
                    }
                )

            active = states
            while active:
                batch_states = []
                batch_payload = []
                for state in active:
                    task = state["task"]
                    comments = state["comments"]
                    attempts = int(state["attempts"])
                    if attempts >= min(budget, len(comments)):
                        continue
                    if _is_online_calibrated_qql(strategy):
                        comment_index = _select_online_calibrated_qql_comment(
                            task,
                            comments,
                            state["queried_indices"],
                            state["online_outcomes"],
                            budget,
                            seed,
                            strategy=strategy,
                            early_feedback=_is_early_feedback_qql(strategy),
                            global_history_rows=global_signal_rows if _is_global_feedback_qql(strategy) else None,
                            global_history_utilities=global_signal_utilities if _is_global_feedback_qql(strategy) else None,
                        )
                    else:
                        comment_index = attempts
                    comment = comments[int(comment_index)]
                    prompt_text = _prompt(task, comment)
                    cache_key = _completion_cache_key(
                        target_name=target.name,
                        model=model,
                        prompt_text=prompt_text,
                        max_new_tokens=os.environ.get("HF_MAX_NEW_TOKENS", ""),
                    )
                    completion = completion_cache.get(cache_key)
                    batch_states.append((state, int(comment_index), comment, prompt_text, cache_key, completion))
                    if completion is None:
                        batch_payload.append((task, prompt_text))

                generated_iter = iter(target.complete_many(batch_payload)) if batch_payload else iter(())
                next_active = []
                for state, comment_index, comment, prompt_text, cache_key, cached_completion in batch_states:
                    task = state["task"]
                    completion = cached_completion
                    if completion is None:
                        completion = next(generated_iter)
                        completion_cache[cache_key] = completion
                        _append_csv(
                            cache_path,
                            {
                                "cache_key": cache_key,
                                "dataset": dataset,
                                "model": model,
                                "target_name": target.name,
                                "max_new_tokens": os.environ.get("HF_MAX_NEW_TOKENS", ""),
                                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                                "completion": completion,
                            },
                        )
                    state["attempts"] = int(state["attempts"]) + 1
                    attempts = int(state["attempts"])
                    if _is_online_calibrated_qql(strategy):
                        state["queried_indices"].append(comment_index)
                    vulnerable = is_vulnerable_completion(task, completion)
                    functional = looks_functional(completion)
                    effective = vulnerable and functional
                    if _is_online_calibrated_qql(strategy):
                        state["online_outcomes"].append((vulnerable, functional, effective))
                        if _is_global_feedback_qql(strategy):
                            signal_rows, _, _ = _comment_quantum_signals(task, state["comments"], seed)
                            global_signal_rows.append(signal_rows[int(comment_index)])
                            global_signal_utilities.append(_query_utility(vulnerable, functional, effective))
                    completion_path = _write_completion(
                        completion_dir,
                        target_arg,
                        strategy,
                        budget,
                        state["task_uid"],
                        attempts,
                        completion,
                    )
                    generated_path = _write_completion(
                        completion_dir,
                        target_arg,
                        strategy,
                        budget,
                        state["task_uid"] + "_full",
                        attempts,
                        prompt_text + completion,
                    )
                    _append_csv(
                        detail_path,
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "target_name": target.name,
                            "budget": budget,
                            "strategy": strategy,
                            "task_index": state["task_index"],
                            "task_id": task.task_id,
                            "language": task.language,
                            "cwe": task.cwe,
                            "attempt": attempts,
                            "comment_id": ATTACK_COMMENTS.index(comment) if comment in ATTACK_COMMENTS else -1,
                            "comment_index": comment_index,
                            "candidate_pool_size": state["candidate_pool_size"],
                            "comment": comment,
                            "vulnerable": int(vulnerable),
                            "functional": int(functional),
                            "effective_vulnerable": int(effective),
                            "completion_path": str(completion_path),
                            "generated_code_path": str(generated_path),
                            "completion_preview": _preview(completion),
                        },
                    )
                    state["best_vulnerable"] = bool(state["best_vulnerable"] or vulnerable)
                    state["best_functional"] = bool(state["best_functional"] or functional)
                    state["best_effective"] = bool(state["best_effective"] or effective)
                    if effective and int(state["attempt_to_success"]) == 0:
                        state["attempt_to_success"] = attempts
                    done = bool(state["best_effective"]) or attempts >= min(budget, len(state["comments"]))
                    if done:
                        _append_csv(
                            outcome_path,
                            {
                                "dataset": dataset,
                                "model": model,
                                "seed": seed,
                                "budget": budget,
                                "strategy": strategy,
                                "task_index": state["task_index"],
                                "task_id": task.task_id,
                                "language": task.language,
                                "cwe": task.cwe,
                                "attempts": attempts,
                                "candidate_pool_size": state["candidate_pool_size"],
                                "vulnerable": int(state["best_vulnerable"]),
                                "functional": int(state["best_functional"]),
                                "effective_vulnerable": int(state["best_effective"]),
                                "attempt_to_success": int(state["attempt_to_success"]),
                            },
                        )
                        completed.add(state["task_key"])
                    else:
                        next_active.append(state)
                active = next_active
                _write_summary(outcome_path, summary_path, total_tasks=total_tasks)
            _write_summary(outcome_path, summary_path, total_tasks=total_tasks)
    _write_summary(outcome_path, summary_path, total_tasks=total_tasks)


def _write_summary(outcome_path: Path, summary_path: Path, *, total_tasks: int) -> None:
    rows = _read_csv(outcome_path) if outcome_path.exists() else []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["seed"], row["budget"], row["strategy"])].append(row)
    summary = []
    for (dataset, model, seed, budget, strategy), items in sorted(grouped.items()):
        completed_tasks = len(items)
        queries = sum(int(row["attempts"]) for row in items)
        vulnerable = sum(int(row["vulnerable"]) for row in items)
        functional = sum(int(row["functional"]) for row in items)
        effective = sum(int(row["effective_vulnerable"]) for row in items)
        first_success = [
            int(row["attempt_to_success"])
            for row in items
            if int(row["attempt_to_success"]) > 0
        ]
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "budget": budget,
                "strategy": strategy,
                "completed_tasks": completed_tasks,
                "total_tasks": total_tasks,
                "is_complete": int(completed_tasks == total_tasks),
                "queries": queries,
                "queries_per_completed_task": queries / max(completed_tasks, 1),
                "queries_per_effective_success": queries / max(effective, 1),
                "q_at_success_completed": fmean(first_success) if first_success else 0.0,
                "unsafe_and_functional_at_q": effective / max(completed_tasks, 1),
                "raw_asr_completed": vulnerable / max(completed_tasks, 1),
                "functional_rate_completed": functional / max(completed_tasks, 1),
                "effective_asr_completed": effective / max(completed_tasks, 1),
            }
        )
    _write_csv(summary_path, summary)


def _merge_summaries(root: Path) -> None:
    rows = []
    for path in root.glob("*/streaming_summary.csv"):
        rows.extend(_read_csv(path))
    _write_csv(root / "streaming_merged_summary.csv", rows)
    if rows:
        agg = _aggregate_rows(rows)
        _write_csv(root / "streaming_merged_summary_agg.csv", agg)


def _aggregate_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["budget"], row["strategy"])].append(row)
    out = []
    for (dataset, model, budget, strategy), items in sorted(grouped.items()):
        complete_items = [row for row in items if int(row["is_complete"])]
        source = complete_items if complete_items else items
        asrs = [float(row["effective_asr_completed"]) for row in source]
        queries = [float(row["queries"]) for row in source]
        qpt = [float(row["queries_per_completed_task"]) for row in source]
        q_success = [float(row.get("q_at_success_completed", "0") or 0.0) for row in source]
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "budget": budget,
                "strategy": strategy,
                "complete_seeds": ",".join(row["seed"] for row in complete_items),
                "all_seeds": ",".join(row["seed"] for row in items),
                "mean_effective_asr": fmean(asrs),
                "mean_queries": fmean(queries),
                "mean_queries_per_task": fmean(qpt),
                "mean_q_at_success": fmean(q_success),
                "mean_unsafe_and_functional_at_q": fmean(asrs),
                "min_completed_tasks": min(int(row["completed_tasks"]) for row in source),
                "max_completed_tasks": max(int(row["completed_tasks"]) for row in source),
            }
        )
    return out


def _completion_cache_key(*, target_name: str, model: str, prompt_text: str, max_new_tokens: str) -> str:
    payload = "\0".join([target_name, model, max_new_tokens, prompt_text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_completion_cache(path: Path) -> dict[str, str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    cache: dict[str, str] = {}
    for row in _read_csv(path):
        key = row.get("cache_key", "")
        if key and "completion" in row:
            cache[key] = row["completion"]
    return cache


def _append_csv(path: Path, row: dict[str, object]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


if __name__ == "__main__":
    main()
