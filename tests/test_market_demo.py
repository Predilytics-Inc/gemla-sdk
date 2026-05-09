import numpy as np

from gemla.data import make_market_microstructure
from gemla.pipelines import GemlaPipeline
from gemla.reports import export_markdown_report


def test_market_microstructure_shape_and_finiteness():
    X = make_market_microstructure(n=500)

    assert X.shape == (500, 5)
    assert np.all(np.isfinite(X))


def test_market_pipeline_passes():
    X = make_market_microstructure()

    result = GemlaPipeline().fit_evaluate(X)

    assert result.verdict is True
    assert result.gate_result["main"]["pass"] is True
    assert result.gate_result["all_controls_rejected"] is True
    assert result.anchor_result["anchor_count"] >= 3
    assert result.winding_result["jump_count"] > 0


def test_market_report_export(tmp_path):
    X = make_market_microstructure()
    result = GemlaPipeline().fit_evaluate(X)

    report_path = export_markdown_report(
        result,
        output_path=tmp_path / "market_report.md",
        title="GEMLA Market Microstructure Report",
    )

    assert report_path.exists()

    text = report_path.read_text(encoding="utf-8")

    assert "GEMLA Market Microstructure Report" in text
    assert "Final Verdict" in text
    assert "Controls" in text
    assert "Winding Diagnostics" in text