import numpy as np


def winding_summary(theta: np.ndarray) -> dict:
    """
    Compute integer winding index and jump summary.
    """
    theta = np.asarray(theta, dtype=float)

    k = np.floor((theta - theta[0]) / np.pi).astype(int)
    jumps = np.where(np.diff(k) != 0)[0] + 1

    if len(jumps) < 2:
        jump_gap_cv = 0.0
    else:
        gaps = np.diff(jumps)
        jump_gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-12))

    return {
        "k_min": int(np.min(k)),
        "k_max": int(np.max(k)),
        "total_winding_cells": int(np.max(k) - np.min(k)),
        "jump_count": int(len(jumps)),
        "jump_gap_cv": jump_gap_cv,
        "k": k,
        "jumps": jumps,
    }