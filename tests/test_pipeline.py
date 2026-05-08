from gemla.data import make_synthetic_transport
from gemla.pipelines import GemlaPipeline


def test_minimal_pipeline_passes():
    X = make_synthetic_transport()

    pipe = GemlaPipeline()
    result = pipe.fit_evaluate(X)

    assert result.verdict is True
    assert result.gate_result["main"]["pass"] is True
    assert result.gate_result["all_controls_rejected"] is True
    assert result.anchor_result["anchor_count"] >= 3
    assert result.anchor_result["spacing_cv"] > 0.01
    assert result.winding_result["jump_count"] > 0