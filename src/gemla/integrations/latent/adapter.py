from __future__ import annotations

import numpy as np


def latent_to_complex_surrogate(
    latents: np.ndarray,
    component_a: int = 0,
    component_b: int = 1,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert a latent embedding trajectory into a complex surrogate trajectory.

    Parameters
    ----------
    latents:
        Array of shape (n_steps, latent_dim).
    component_a:
        First latent component used as real part.
    component_b:
        Second latent component used as imaginary part.
    normalize:
        Whether to standardize selected components before conversion.

    Returns
    -------
    z:
        Complex surrogate trajectory of shape (n_steps,).
    """
    latents = np.asarray(latents, dtype=float)

    if latents.ndim != 2:
        raise ValueError("latents must have shape (n_steps, latent_dim).")

    if latents.shape[1] < 2:
        raise ValueError("latents must have at least two dimensions.")

    if component_a >= latents.shape[1] or component_b >= latents.shape[1]:
        raise ValueError("component indices exceed latent dimension.")

    x = latents[:, component_a]
    y = latents[:, component_b]

    if normalize:
        eps = 1e-12
        x = (x - np.mean(x)) / (np.std(x) + eps)
        y = (y - np.mean(y)) / (np.std(y) + eps)

    z = x + 1j * y

    eps = 1e-12
    z = np.where(np.abs(z) < eps, z + eps, z)

    return z


def make_synthetic_latents(
    n: int = 1200,
    latent_dim: int = 16,
    t_max: float = 80.0,
    noise: float = 0.01,
    seed: int = 17,
) -> np.ndarray:
    """
    Generate a synthetic latent embedding trajectory with coherent transport.

    This simulates what an external video/world-model encoder might output:
    a sequence of latent vectors with smooth transport structure plus noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n)

    theta = 0.50 * t + 0.15 * np.sin(0.7 * t) + 0.05 * np.sin(1.9 * t)

    latents = noise * rng.normal(size=(n, latent_dim))

    latents[:, 0] += np.cos(theta)
    latents[:, 1] += np.sin(theta)

    for j in range(2, latent_dim):
        latents[:, j] += 0.2 * np.sin((0.05 * j + 0.1) * t + 0.3 * j)

    return latents