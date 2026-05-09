from gemla.cli.main import main


def test_cli_evaluate_vjepa_synthetic_runs(tmp_path, monkeypatch, capsys):
    synthetic_output = tmp_path / "sample_vjepa_like_embeddings.npy"

    monkeypatch.setattr(
        "sys.argv",
        [
            "gemla",
            "evaluate-vjepa",
            "--synthetic",
            "--synthetic-output",
            str(synthetic_output),
            "--n",
            "400",
            "--latent-dim",
            "12",
        ],
    )

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "GEMLA V-JEPA-Style Latent Transport Evaluation" in captured.out
    assert "Input embeddings:" in captured.out
    assert "GEMLA Latent Transport Evaluation" in captured.out
    assert "Final verdict: PASS" in captured.out
    assert synthetic_output.exists()