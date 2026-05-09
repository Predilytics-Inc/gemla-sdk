import numpy as np

from gemla.integrations.vjepa import (
    load_vjepa_embeddings,
    save_sample_vjepa_like_embeddings,
    vjepa_embeddings_to_complex_surrogate,
)
from gemla.pipelines import GemlaLatentPipeline


def test_save_and_load_sample_vjepa_like_embeddings(tmp_path):
    path = tmp_path / "sample_embeddings.npy"

    saved_path = save_sample_vjepa_like_embeddings(
        path,
        n=300,
        latent_dim=12,
    )

    embeddings = load_vjepa_embeddings(saved_path)

    assert embeddings.shape == (300, 12)
    assert np.all(np.isfinite(embeddings))


def test_vjepa_embeddings_to_complex_surrogate(tmp_path):
    path = tmp_path / "sample_embeddings.npy"
    save_sample_vjepa_like_embeddings(path, n=300, latent_dim=12)

    embeddings = load_vjepa_embeddings(path)
    z = vjepa_embeddings_to_complex_surrogate(embeddings)

    assert z.shape == (300,)
    assert np.iscomplexobj(z)
    assert np.all(np.isfinite(z.real))
    assert np.all(np.isfinite(z.imag))


def test_vjepa_style_pipeline_passes(tmp_path):
    path = tmp_path / "sample_embeddings.npy"
    save_sample_vjepa_like_embeddings(path)

    embeddings = load_vjepa_embeddings(path)
    result = GemlaLatentPipeline().fit_evaluate(embeddings)

    assert result.verdict is True
    assert result.gate_result["main"]["pass"] is True
    assert result.gate_result["all_controls_rejected"] is True
    assert result.anchor_result["anchor_count"] >= 3
    assert result.winding_result["jump_count"] > 0