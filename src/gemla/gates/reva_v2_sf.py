import numpy as np

from gemla.lifted.lifted_phase import lifted_phase, phase_derivatives
from gemla.lifted.spectral_flatness import spectral_flatness


def reva_v2_metrics(theta: np.ndarray) -> dict:
    """
    Compute RevA v2 lifted-phase metrics.
    """
    theta = np.asarray(theta, dtype=float)
    dtheta, d2theta = phase_derivatives(theta)

    eps = 1e-12
    signs = np.sign(dtheta)
    sign_changes = int(np.sum(np.diff(signs) != 0))

    return {
        "phase_growth": float(theta[-1] - theta[0]),
        "orientation": float(np.mean(signs)),
        "smoothness": float(np.std(dtheta) / (np.mean(np.abs(dtheta)) + eps)),
        "curvature": float(np.std(d2theta)),
        "sign_changes": sign_changes,
        "sign_change_ratio": float(sign_changes / max(len(theta) - 1, 1)),
    }


def hard_pass(metrics: dict) -> bool:
    """
    Hard RevA v2 constraints.
    """
    return (
        metrics["phase_growth"] > 1e-3
        and metrics["orientation"] > 0.60
        and metrics["sign_change_ratio"] < 0.10
        and metrics["smoothness"] < 1.50
    )


def evaluate_reva_v2_sf(
    main_z: np.ndarray,
    control_zs: dict[str, np.ndarray],
    flatness_max: float = 1.25,
) -> dict:
    """
    Evaluate main branch and controls using RevA v2 + spectral flatness.
    """
    main_theta = lifted_phase(main_z)
    main_metrics = reva_v2_metrics(main_theta)
    main_flatness = spectral_flatness(main_theta)

    main_pass = hard_pass(main_metrics) and main_flatness["score"] < flatness_max

    controls = {}
    for name, zc in control_zs.items():
        theta_c = lifted_phase(zc)
        metrics_c = reva_v2_metrics(theta_c)
        flatness_c = spectral_flatness(theta_c)

        control_pass = hard_pass(metrics_c) and flatness_c["score"] < flatness_max

        controls[name] = {
            "metrics": metrics_c,
            "flatness": flatness_c,
            "pass": bool(control_pass),
            "rejected": bool(not control_pass),
        }

    all_controls_rejected = all(c["rejected"] for c in controls.values())

    return {
        "main": {
            "metrics": main_metrics,
            "flatness": main_flatness,
            "pass": bool(main_pass),
        },
        "controls": controls,
        "all_controls_rejected": bool(all_controls_rejected),
        "pass": bool(main_pass and all_controls_rejected),
    }