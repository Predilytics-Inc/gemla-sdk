from __future__ import annotations

import numpy as np


def make_cyber_event_transport(
    n: int = 1200,
    t_max: float = 80.0,
    noise: float = 0.014,
    seed: int = 53,
) -> np.ndarray:
    """
    Generate synthetic cybersecurity event telemetry with coherent campaign-like transport.

    Columns simulate:
    - identity/authentication pressure phase component
    - lateral/network movement phase component
    - privilege escalation proxy
    - DNS / command-and-control proxy
    - exfiltration / staging proxy

    The first two columns are suitable for GEMLA's complex surrogate:
        z(t) = X[:, 0] + i X[:, 1].
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n)

    # Coherent campaign phase: weak progression with staged modulation.
    theta = (
        0.46 * t
        + 0.20 * np.sin(0.34 * t)
        + 0.07 * np.sin(1.25 * t)
    )

    # Campaign envelope activates around mid-horizon.
    campaign = 1.0 + 0.12 / (1.0 + np.exp(-(t - 43.0) / 4.5))

    auth_pressure = campaign * np.cos(theta) + noise * rng.normal(size=n)
    lateral_movement = campaign * np.sin(theta) + noise * rng.normal(size=n)

    privilege_escalation = (
        0.10
        + 0.05 * np.maximum(campaign - 1.0, 0.0)
        + 0.03 * np.sin(0.19 * t + 0.4)
        + noise * rng.normal(size=n)
    )

    dns_c2_proxy = (
        0.08
        + 0.04 * np.sin(0.29 * t + theta / 10.0)
        + 0.06 * np.maximum(campaign - 1.0, 0.0)
        + noise * rng.normal(size=n)
    )

    exfiltration_staging = (
        0.05
        + 0.12 * np.maximum(t - 48.0, 0.0) / max(t_max - 48.0, 1.0)
        + 0.025 * np.sin(0.41 * t)
        + noise * rng.normal(size=n)
    )

    return np.column_stack(
        [
            auth_pressure,
            lateral_movement,
            privilege_escalation,
            dns_c2_proxy,
            exfiltration_staging,
        ]
    )