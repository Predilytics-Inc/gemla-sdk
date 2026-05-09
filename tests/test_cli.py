from gemla.cli.main import main


def test_cli_evaluate_runs(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "cli_report.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "gemla",
            "evaluate",
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GEMLA Transport Evaluation" in captured.out
    assert "Final verdict: PASS" in captured.out
    assert "Report written to:" in captured.out
    assert output_path.exists()

    report_text = output_path.read_text(encoding="utf-8")

    assert "GEMLA Transport Evaluation Report" in report_text
    assert "Final Verdict" in report_text
    assert "Controls" in report_text
    assert "Winding Diagnostics" in report_text