import numpy as np


def make_synthetic_transport(
    n: int = 1200,
    t_max: float = 80.0,
    noise: float = 0.01,
    seed: int = 7,
) -> np.ndarray:
    """
    Generate a simple coherent transport signal.

    Returns
    -------
    X : np.ndarray, shape (n, 3)
        Columns:
        - cos(theta)
        - sin(theta)
        - auxiliary amplitude channel
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n)

    theta = (
        0.55 * t
        + 0.18 * np.sin(0.9 * t)
        + 0.07 * np.sin(2.1 * t)
    )

    x0 = np.cos(theta) + noise * rng.normal(size=n)
    x1 = np.sin(theta) + noise * rng.normal(size=n)
    x2 = 1.0 + 0.15 * np.sin(0.31 * t) + noise * rng.normal(size=n)

    return np.column_stack([x0, x1, x2])