"""Build a compact raw-trace artifact from local streaming outputs.

The artifact intentionally avoids committing hundreds of megabytes of generated
code.  It commits the per-task detector outcomes needed to reaggregate the
paper tables, plus hashes of source CSV/cache files and completion payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DEFAULT_MAIN_ROOTS = [
    "outputs/ccfa_protocol_securityeval_qwen05_seed7_full_gatefix_20260706",
    "outputs/ccfa_protocol_llmseceval_qwen05_seed7_full_gatefix_20260706",
    "outputs/ccfa_protocol_securityeval_qwen15_5seed_full_gatefix_20260707",
    "outputs/ccfa_protocol_llmseceval_qwen15_2method_5seed_batched_20260707",
    "outputs/ccfa_protocol_llmseceval_qwen15_strong_baselines_20260708",
]
DEFAULT_CYBER_ROOTS = [
    "outputs/ccfa_protocol_cyberseceval_qwen05_120_5seed_strong_v2_20260708",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-roots", default=",".join(DEFAULT_MAIN_ROOTS))
    parser.add_argument("--cyber-roots", default=",".join(DEFAULT_CYBER_ROOTS))
    parser.add_argument("--output-dir", default="paper_artifacts/ccfa_trace_20260708")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_roots = _split_roots(args.main_roots)
    cyber_roots = _split_roots(args.cyber_roots)

    main_outcomes = _collect_task_outcomes(main_roots)
    cyber_outcomes = _collect_task_outcomes(cyber_roots)
    source_hashes = _source_hashes(main_roots + cyber_roots)
    cache_manifest = _completion_cache_manifest(main_roots + cyber_roots)

    _write_csv(out_dir / "main_task_outcomes.csv", main_outcomes)
    _write_csv(out_dir / "cyberseceval_task_outcomes.csv", cyber_outcomes)
    _write_csv(out_dir / "source_file_hashes.csv", source_hashes)
    _write_csv(out_dir / "completion_cache_manifest.csv", cache_manifest)

    manifest = {
        "main_roots": [str(path) for path in main_roots],
        "cyber_roots": [str(path) for path in cyber_roots],
        "main_task_outcome_rows": len(main_outcomes),
        "cyberseceval_task_outcome_rows": len(cyber_outcomes),
        "source_hash_rows": len(source_hashes),
        "completion_cache_rows": len(cache_manifest),
        "artifact_policy": "compact trace ledger with hashes, not full generated-code payloads",
    }
    (out_dir / "trace_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "TRACE_MANIFEST.md").write_text(_manifest_markdown(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def _split_roots(raw: str) -> list[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def _collect_task_outcomes(roots: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for root in roots:
        for path in sorted(root.glob("*/streaming_task_outcomes.csv")):
            for row in _read_csv(path):
                kept = {
                    "source_root": str(root).replace("\\", "/"),
                    "source_file_sha256": _sha256_file(path),
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "seed": row["seed"],
                    "budget": row["budget"],
                    "strategy": row["strategy"],
                    "task_index": row["task_index"],
                    "task_id": row["task_id"],
                    "language": row.get("language", ""),
                    "cwe": row.get("cwe", ""),
                    "attempts": row["attempts"],
                    "candidate_pool_size": row.get("candidate_pool_size", ""),
                    "vulnerable": row["vulnerable"],
                    "functional": row["functional"],
                    "effective_vulnerable": row["effective_vulnerable"],
                    "attempt_to_success": row.get("attempt_to_success", "0"),
                }
                kept["task_key_sha256"] = _sha256_text(
                    "|".join(
                        [
                            kept["dataset"],
                            kept["model"],
                            kept["seed"],
                            kept["budget"],
                            kept["strategy"],
                            kept["task_id"],
                        ]
                    )
                )
                rows.append(kept)
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["model"],
            int(row["budget"]),
            row["strategy"],
            int(row["seed"]),
            int(row["task_index"]),
        )
    )
    return rows


def _source_hashes(roots: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    patterns = [
        "completion_cache.csv",
        "streaming_merged_summary.csv",
        "streaming_merged_summary_agg.csv",
        "*/streaming_summary.csv",
        "*/streaming_task_outcomes.csv",
        "*/streaming_detail.csv",
    ]
    seen: set[Path] = set()
    for root in roots:
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                rows.append(
                    {
                        "path": str(path).replace("\\", "/"),
                        "bytes": str(path.stat().st_size),
                        "sha256": _sha256_file(path),
                    }
                )
    return rows


def _completion_cache_manifest(roots: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for root in roots:
        path = root / "completion_cache.csv"
        if not path.exists():
            continue
        for row in _read_csv(path):
            completion = row.get("completion", "")
            rows.append(
                {
                    "source_root": str(root).replace("\\", "/"),
                    "cache_key": row.get("cache_key", ""),
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "target_name": row.get("target_name", ""),
                    "max_new_tokens": row.get("max_new_tokens", ""),
                    "prompt_sha256": row.get("prompt_sha256", ""),
                    "completion_sha256": _sha256_text(completion),
                    "completion_bytes_utf8": str(len(completion.encode("utf-8"))),
                }
            )
    rows.sort(key=lambda row: (row["dataset"], row["model"], row["cache_key"]))
    return rows


def _manifest_markdown(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "# QScout CCF-A Compact Trace Artifact",
            "",
            "This artifact is the committed raw-trace verification layer for the",
            "CCF-A evidence package.  It does not commit full generated code or",
            "model caches; instead it commits detector outcomes and hashes that",
            "allow table reaggregation and local raw-output integrity checks.",
            "",
            f"- Main task-outcome rows: {manifest['main_task_outcome_rows']}",
            f"- CyberSecEval task-outcome rows: {manifest['cyberseceval_task_outcome_rows']}",
            f"- Source file hash rows: {manifest['source_hash_rows']}",
            f"- Completion cache hash rows: {manifest['completion_cache_rows']}",
            "",
            "Run:",
            "",
            "```powershell",
            '& "D:\\ProgramData\\py2\\python.exe" scripts\\verify_trace_artifact.py',
            "```",
            "",
        ]
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
