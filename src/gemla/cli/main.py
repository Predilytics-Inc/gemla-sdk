from __future__ import annotations

import argparse
from pathlib import Path

from gemla.benchmarks import (
    run_benchmark_suite,
    summarize_benchmark_records,
    write_benchmark_results,
)
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

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run the GEMLA benchmark suite.",
    )

    benchmark.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Directory where benchmark CSV/JSON outputs will be written.",
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


def run_benchmark(args: argparse.Namespace) -> int:
    records = run_benchmark_suite()
    summary = summarize_benchmark_records(records)
    paths = write_benchmark_results(
        records,
        output_dir=Path(args.output_dir),
    )

    print("GEMLA Benchmark Suite")
    print("---------------------")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Passed cases: {summary['passed_cases']}")
    print(f"Failed cases: {summary['failed_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"All wrong-sign rejected: {summary['all_wrong_sign_rejected']}")
    print(f"All phase-shuffle rejected: {summary['all_phase_shuffle_rejected']}")
    print(f"All residue-scramble rejected: {summary['all_residue_scramble_rejected']}")
    print(f"Minimum anchor count: {summary['min_anchor_count']}")
    print(f"Minimum winding jumps: {summary['min_winding_jumps']}")
    print()
    print(f"CSV written to: {paths['csv']}")
    print(f"JSON written to: {paths['json']}")
    print(f"Summary written to: {paths['summary']}")

    return 0 if summary["all_passed"] else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        return run_evaluate(args)

    if args.command == "benchmark":
        return run_benchmark(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())