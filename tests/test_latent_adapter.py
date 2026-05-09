import numpy as np

from gemla.integrations.latent import latent_to_complex_surrogate, make_synthetic_latents
from gemla.pipelines import GemlaLatentPipeline


def test_synthetic_latents_shape():
    latents = make_synthetic_latents(n=200, latent_dim=8)

    assert latents.shape == (200, 8)
    assert np.all(np.isfinite(latents))


def test_latent_to_complex_surrogate():
    latents = make_synthetic_latents(n=200, latent_dim=8)
    z = latent_to_complex_surrogate(latents)

    assert z.shape == (200,)
    assert np.iscomplexobj(z)
    assert np.all(np.isfinite(z.real))
    assert np.all(np.isfinite(z.imag))


def test_latent_pipeline_passes():
    latents = make_synthetic_latents()
    result = GemlaLatentPipeline().fit_evaluate(latents)

    assert result.verdict is True
    assert result.gate_result["main"]["pass"] is True
    assert result.gate_result["all_controls_rejected"] is True
    assert result.anchor_result["anchor_count"] >= 3
    assert result.winding_result["jump_count"] > 0