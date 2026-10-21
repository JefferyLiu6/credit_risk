"""Small command-line interface for repeatable project operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import build_validation_report, write_validation_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="credit-risk")
    subcommands = parser.add_subparsers(dest="command", required=True)
    report = subcommands.add_parser("validation-report", help="rebuild model comparison report")
    report.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/notebook3_validation_predictions.csv"),
    )
    report.add_argument(
        "--output", type=Path, default=Path("reports/validation_decision_generated.json")
    )
    report.add_argument("--resamples", type=int, default=500)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validation-report":
        report = build_validation_report(args.predictions, n_resamples=args.resamples)
        write_validation_report(report, args.output)
        summary = {
            "incumbent": report["incumbent_champion"],
            "candidate": report["recommended_candidate"],
            "output": str(args.output),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
