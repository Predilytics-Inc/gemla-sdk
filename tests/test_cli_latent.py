from gemla.cli.main import main


def test_cli_evaluate_latent_runs(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "gemla",
            "evaluate-latent",
            "--n",
            "400",
            "--latent-dim",
            "8",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GEMLA Latent Transport Evaluation" in captured.out
    assert "Input shape:" in captured.out
    assert "Final verdict: PASS" in captured.out