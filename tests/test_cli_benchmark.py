from gemla.cli.main import main


def test_cli_benchmark_runs(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "benchmark_results"

    monkeypatch.setattr(
        "sys.argv",
        [
            "gemla",
            "benchmark",
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GEMLA Benchmark Suite" in captured.out
    assert "Total cases:" in captured.out
    assert "Pass rate:" in captured.out
    assert "CSV written to:" in captured.out
    assert "JSON written to:" in captured.out
    assert "Summary written to:" in captured.out

    assert (output_dir / "benchmark_results.csv").exists()
    assert (output_dir / "benchmark_results.json").exists()
    assert (output_dir / "benchmark_summary.json").exists()