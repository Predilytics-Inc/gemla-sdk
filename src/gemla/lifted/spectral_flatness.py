import numpy as np


def spectral_flatness(theta: np.ndarray, lambda_curv: float = 0.05) -> dict:
    """
    Compute a simple spectral-flatness proxy from lifted phase.
    """
    theta = np.asarray(theta, dtype=float)

    dtheta = np.gradient(theta)
    d2theta = np.gradient(dtheta)

    eps = 1e-12
    rho = dtheta / np.pi
    rho_norm = (rho - np.mean(rho)) / (np.std(rho) + eps)

    variance_part = float(np.var(rho_norm))
    curvature_part = float(lambda_curv * np.mean(d2theta ** 2))
    score = variance_part + curvature_part

    return {
        "score": float(score),
        "variance_part": variance_part,
        "curvature_part": curvature_part,
    }