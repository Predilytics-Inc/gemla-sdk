from gemla.benchmarks import (
    run_benchmark_suite,
    summarize_benchmark_records,
    write_benchmark_results,
)


def main() -> None:
    records = run_benchmark_suite()
    summary = summarize_benchmark_records(records)
    paths = write_benchmark_results(records)

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


if __name__ == "__main__":
    main()