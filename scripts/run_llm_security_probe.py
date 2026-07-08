"""Reviewer-friendly entry point for QScout-QBW LLM security probing."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STRATEGIES_FULL = (
    "fair_random_comment,"
    "fair_risk_prior_comment,"
    "classical_active_comment,"
    "insec_fixed_pool_comment,"
    "aot_ensemble_fixed_pool_comment,"
    "classical_boundary_witness_comment,"
    "qscout_qbw_comment"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or export the QScout-QBW hard-label LLM code-security protocol."
    )
    parser.add_argument("--dataset", choices=["securityeval", "llmseceval", "cyberseceval"], default="securityeval")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--budgets", default="4,8,16")
    parser.add_argument("--seeds", default="7,19,31,43,59")
    parser.add_argument("--strategies", default=STRATEGIES_FULL)
    parser.add_argument("--target", choices=["hf", "offline", "openai", "gemini"], default="hf")
    parser.add_argument("--prompt-mode", choices=["auto", "raw", "instruction"], default="instruction")
    parser.add_argument("--max-new-tokens", default="80")
    parser.add_argument("--batch-size", default="8")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--export-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or _default_output_dir(args.dataset, args.model)
    cmd = [
        sys.executable,
        "run_llm_topconf_streaming_matrix.py",
        "--target",
        args.target,
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--strategies",
        args.strategies,
        "--budgets",
        args.budgets,
        "--seeds",
        args.seeds,
        "--prompt-mode",
        args.prompt_mode,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--batch-size",
        str(args.batch_size),
        "--output-dir",
        output_dir,
    ]
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])

    if args.dry_run:
        print(" ".join(_quote(part) for part in cmd))
    else:
        subprocess.run(cmd, check=True)

    if args.export_results:
        export_cmd = [sys.executable, "scripts/export_llm_security_results.py"]
        if args.dry_run:
            print(" ".join(_quote(part) for part in export_cmd))
        else:
            subprocess.run(export_cmd, check=True)


def _default_output_dir(dataset: str, model: str) -> str:
    clean_model = model.replace("/", "_").replace(":", "_")
    return str(Path("outputs") / f"llm_security_probe_{dataset}_{clean_model}")


def _quote(value: str) -> str:
    if any(char.isspace() for char in value):
        return '"' + value.replace('"', '\\"') + '"'
    return value


if __name__ == "__main__":
    main()
