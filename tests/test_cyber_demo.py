import numpy as np

from gemla.data import make_cyber_event_transport
from gemla.pipelines import GemlaPipeline
from gemla.reports import export_markdown_report


def test_cyber_event_transport_shape_and_finiteness():
    X = make_cyber_event_transport(n=500)

    assert X.shape == (500, 5)
    assert np.all(np.isfinite(X))


def test_cyber_pipeline_passes():
    X = make_cyber_event_transport()

    result = GemlaPipeline().fit_evaluate(X)

    assert result.verdict is True
    assert result.gate_result["main"]["pass"] is True
    assert result.gate_result["all_controls_rejected"] is True
    assert result.anchor_result["anchor_count"] >= 3
    assert result.winding_result["jump_count"] > 0


def test_cyber_report_export(tmp_path):
    X = make_cyber_event_transport()
    result = GemlaPipeline().fit_evaluate(X)

    report_path = export_markdown_report(
        result,
        output_path=tmp_path / "cyber_report.md",
        title="GEMLA Cyber Event Transport Report",
    )

    assert report_path.exists()

    text = report_path.read_text(encoding="utf-8")

    assert "GEMLA Cyber Event Transport Report" in text
    assert "Final Verdict" in text
    assert "Controls" in text
    assert "Winding Diagnostics" in text