from gemla.cli.main import main


def test_cli_demo_industrial_runs(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "industrial_report.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "gemla",
            "demo-industrial",
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GEMLA Industrial Telemetry Demo" in captured.out
    assert "Final verdict: PASS" in captured.out
    assert output_path.exists()


def test_cli_demo_market_runs(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "market_report.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "gemla",
            "demo-market",
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GEMLA Market Microstructure Demo" in captured.out
    assert "Final verdict: PASS" in captured.out
    assert output_path.exists()


def test_cli_demo_cyber_runs(tmp_path, monkeypatch, capsys):
    output_path = tmp_path / "cyber_report.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "gemla",
            "demo-cyber",
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GEMLA Cyber Event Transport Demo" in captured.out
    assert "Final verdict: PASS" in captured.out
    assert output_path.exists()