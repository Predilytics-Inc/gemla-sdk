from __future__ import annotations

from pathlib import Path

import numpy as np

from gemla.integrations.latent import latent_to_complex_surrogate


def load_vjepa_embeddings(path: str | Path) -> np.ndarray:
    """
    Load externally extracted V-JEPA-style embeddings.

    Expected shape:
        (n_steps, latent_dim)

    This function does not download or run V-JEPA. It only loads embeddings
    that were already produced by an external encoder.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    if path.suffix != ".npy":
        raise ValueError("Expected a .npy file containing latent embeddings.")

    embeddings = np.load(path)

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected embeddings with shape (n_steps, latent_dim). "
            f"Got shape {embeddings.shape}."
        )

    if embeddings.shape[1] < 2:
        raise ValueError("Embeddings must have at least two latent dimensions.")

    return embeddings.astype(float)


def vjepa_embeddings_to_complex_surrogate(
    embeddings: np.ndarray,
    component_a: int = 0,
    component_b: int = 1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert V-JEPA-style embeddings into a GEMLA complex surrogate trajectory.
    """
    return latent_to_complex_surrogate(
        embeddings,
        component_a=component_a,
        component_b=component_b,
        normalize=normalize,
    )


def save_sample_vjepa_like_embeddings(
    output_path: str | Path,
    n: int = 1200,
    latent_dim: int = 32,
    t_max: float = 80.0,
    noise: float = 0.01,
    seed: int = 23,
) -> Path:
    """
    Save synthetic V-JEPA-like embeddings for demo/testing.

    These are not real V-JEPA embeddings. They are a deterministic synthetic
    stand-in so the integration path can be tested without third-party weights.
    """
    from gemla.integrations.latent import make_synthetic_latents

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = make_synthetic_latents(
        n=n,
        latent_dim=latent_dim,
        t_max=t_max,
        noise=noise,
        seed=seed,
    )

    np.save(output_path, embeddings)
    return output_path