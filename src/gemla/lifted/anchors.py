import numpy as np


def anchor_summary(theta: np.ndarray, quantile: float = 0.90) -> dict:
    """
    Detect curvature anchors and summarize anchor-spacing irregularity.
    """
    theta = np.asarray(theta, dtype=float)

    dtheta = np.gradient(theta)
    d2theta = np.abs(np.gradient(dtheta))

    threshold = np.quantile(d2theta, quantile)

    anchors = []
    for i in range(1, len(d2theta) - 1):
        if d2theta[i] >= threshold and d2theta[i] >= d2theta[i - 1] and d2theta[i] >= d2theta[i + 1]:
            anchors.append(i)

    anchors = np.asarray(anchors, dtype=int)

    if len(anchors) < 3:
        return {
            "anchor_count": int(len(anchors)),
            "spacing_mean": 0.0,
            "spacing_std": 0.0,
            "spacing_cv": 0.0,
            "anchors": anchors,
        }

    spacing = np.diff(anchors)
    spacing_mean = float(np.mean(spacing))
    spacing_std = float(np.std(spacing))
    spacing_cv = float(spacing_std / (spacing_mean + 1e-12))

    return {
        "anchor_count": int(len(anchors)),
        "spacing_mean": spacing_mean,
        "spacing_std": spacing_std,
        "spacing_cv": spacing_cv,
        "anchors": anchors,
    }