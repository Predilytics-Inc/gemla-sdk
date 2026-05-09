from gemla.benchmarks import (
    BenchmarkCase,
    run_benchmark_suite,
    summarize_benchmark_records,
    write_benchmark_results,
)


def test_benchmark_suite_runs_small_set(tmp_path):
    cases = [
        BenchmarkCase(name="baseline_test", n=1200, t_max=80.0, noise=0.01, seed=7),
        BenchmarkCase(name="alt_seed_test", n=1200, t_max=80.0, noise=0.01, seed=13),
    ]

    records = run_benchmark_suite(cases)
    summary = summarize_benchmark_records(records)
    paths = write_benchmark_results(records, output_dir=tmp_path)

    assert len(records) == 2
    assert summary["total_cases"] == 2
    assert summary["all_wrong_sign_rejected"] is True
    assert summary["all_phase_shuffle_rejected"] is True
    assert summary["all_residue_scramble_rejected"] is True
    assert summary["min_anchor_count"] >= 3
    assert summary["min_winding_jumps"] > 0

    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert paths["summary"].exists()