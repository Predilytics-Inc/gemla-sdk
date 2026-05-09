from __future__ import annotations

import numpy as np


def make_market_microstructure(
    n: int = 1200,
    t_max: float = 80.0,
    noise: float = 0.012,
    seed: int = 41,
) -> np.ndarray:
    """
    Generate synthetic market microstructure data with coherent regime transport.

    Columns simulate:
    - order-flow imbalance phase component
    - liquidity/price-pressure phase component
    - spread proxy
    - realized-volatility proxy
    - queue/cancellation pressure proxy

    The first two columns are suitable for GEMLA's complex surrogate:
        z(t) = X[:, 0] + i X[:, 1].
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n)

    # Coherent latent regime phase.
    theta = (
        0.52 * t
        + 0.16 * np.sin(0.38 * t)
        + 0.05 * np.sin(1.7 * t)
    )

    # Regime stress envelope: liquidity pressure increases around mid-horizon.
    stress = 1.0 + 0.10 / (1.0 + np.exp(-(t - 44.0) / 5.0))
    intraday = 1.0 + 0.06 * np.sin(0.10 * t + 0.4)

    order_flow_imbalance = stress * np.cos(theta) + noise * rng.normal(size=n)
    liquidity_pressure = stress * np.sin(theta) + noise * rng.normal(size=n)

    spread_proxy = (
        0.02
        + 0.004 * intraday
        + 0.006 * np.maximum(stress - 1.0, 0.0)
        + noise * 0.05 * rng.normal(size=n)
    )

    realized_vol_proxy = (
        0.10
        + 0.025 * np.abs(np.sin(0.21 * t + theta / 12.0))
        + 0.020 * np.maximum(stress - 1.0, 0.0)
        + noise * rng.normal(size=n)
    )

    cancellation_pressure = (
        0.30
        + 0.05 * np.sin(0.27 * t + 0.8)
        + 0.08 * np.maximum(stress - 1.0, 0.0)
        + noise * rng.normal(size=n)
    )

    return np.column_stack(
        [
            order_flow_imbalance,
            liquidity_pressure,
            spread_proxy,
            realized_vol_proxy,
            cancellation_pressure,
        ]
    )