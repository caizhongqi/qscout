"""QScout project entry point.

Examples:
  python run.py --mode smoke
  python run.py --mode main --figures
  python run.py --figures-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QScout studies and evidence-bound figures.")
    parser.add_argument("--mode", choices=["smoke", "main", "ablations", "noise", "defense", "all"], default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--victim-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=14)
    args = parser.parse_args()
    if not args.figures_only:
        command = [sys.executable, "run_study_matrix.py", "--mode", args.mode, "--victim-epochs", str(args.victim_epochs), "--final-epochs", str(args.final_epochs)]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)
    if args.figures or args.figures_only:
        subprocess.run([sys.executable, "generate_main_result_figures.py"], check=True)


if __name__ == "__main__":
    main()
