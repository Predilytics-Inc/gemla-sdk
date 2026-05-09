from __future__ import annotations

import numpy as np


def make_industrial_telemetry(
    n: int = 1200,
    t_max: float = 80.0,
    noise: float = 0.015,
    seed: int = 31,
) -> np.ndarray:
    """
    Generate synthetic industrial telemetry with coherent transport structure.

    Columns simulate:
    - vibration-like oscillatory state
    - thermal/process phase state
    - pressure/load modulation
    - actuator/load drift
    - fault-propagation proxy

    The first two columns are intentionally suitable for GEMLA's
    complex surrogate: z(t) = X[:, 0] + i X[:, 1].
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n)

    # Coherent latent transport phase: slow drift + modulation.
    theta = (
        0.48 * t
        + 0.18 * np.sin(0.45 * t)
        + 0.06 * np.sin(1.35 * t)
    )

    # Slowly increasing degradation envelope.
    degradation = 1.0 + 0.0025 * t + 0.08 / (1.0 + np.exp(-(t - 45.0) / 4.0))

    vibration = degradation * np.cos(theta) + noise * rng.normal(size=n)
    thermal_phase = degradation * np.sin(theta) + noise * rng.normal(size=n)

    pressure = (
        1.0
        + 0.12 * np.sin(0.18 * t + 0.5)
        + 0.04 * np.sin(0.91 * t)
        + noise * rng.normal(size=n)
    )

    actuator_load = (
        0.7
        + 0.002 * t
        + 0.08 * np.sin(0.11 * t)
        + noise * rng.normal(size=n)
    )

    fault_proxy = (
        0.15 * np.maximum(t - 42.0, 0.0) / max(t_max - 42.0, 1.0)
        + 0.05 * np.sin(0.33 * t + theta / 7.0)
        + noise * rng.normal(size=n)
    )

    return np.column_stack(
        [
            vibration,
            thermal_phase,
            pressure,
            actuator_load,
            fault_proxy,
        ]
    )