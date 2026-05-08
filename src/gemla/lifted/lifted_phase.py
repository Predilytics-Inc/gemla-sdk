import numpy as np


def complex_from_xy(X: np.ndarray) -> np.ndarray:
    """
    Convert first two columns of X into a complex trajectory.
    """
    X = np.asarray(X)

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X must have shape (n_samples, >=2).")

    z = X[:, 0] + 1j * X[:, 1]

    eps = 1e-12
    if np.any(np.abs(z) < eps):
        z = z + eps

    return z


def lifted_phase(z: np.ndarray) -> np.ndarray:
    """
    Compute lifted phase Θ(t) = unwrap(arg(z(t))).
    """
    z = np.asarray(z)

    if z.ndim != 1:
        raise ValueError("z must be a one-dimensional complex trajectory.")

    return np.unwrap(np.angle(z))


def phase_derivatives(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return first and second finite-difference derivatives.
    """
    theta = np.asarray(theta, dtype=float)

    dtheta = np.gradient(theta)
    d2theta = np.gradient(dtheta)

    return dtheta, d2theta