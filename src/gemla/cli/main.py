from __future__ import annotations

import argparse
from pathlib import Path

from gemla.data import make_synthetic_transport
from gemla.pipelines import GemlaPipeline
from gemla.reports import export_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemla",
        description="GEMLA: Γ–EML–α Transport Architecture CLI",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run the minimal GEMLA transport evaluation pipeline.",
    )

    evaluate.add_argument(
        "--output",
        type=str,
        default="reports/gemla_transport_report.md",
        help="Path where the Markdown report will be written.",
    )

    evaluate.add_argument(
        "--n",
        type=int,
        default=1200,
        help="Number of synthetic samples.",
    )

    evaluate.add_argument(
        "--t-max",
        type=float,
        default=80.0,
        help="Maximum synthetic time value.",
    )

    evaluate.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Synthetic noise level.",
    )

    evaluate.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Synthetic data random seed.",
    )

    evaluate.add_argument(
        "--pipeline-seed",
        type=int,
        default=11,
        help="Pipeline/control random seed.",
    )

    return parser


def run_evaluate(args: argparse.Namespace) -> int:
    X = make_synthetic_transport(
        n=args.n,
        t_max=args.t_max,
        noise=args.noise,
        seed=args.seed,
    )

    pipe = GemlaPipeline(seed=args.pipeline_seed)
    result = pipe.fit_evaluate(X)

    print(result.summary())

    report_path = export_markdown_report(
        result,
        output_path=Path(args.output),
    )

    print()
    print(f"Report written to: {report_path}")

    return 0 if result.verdict else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        return run_evaluate(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())