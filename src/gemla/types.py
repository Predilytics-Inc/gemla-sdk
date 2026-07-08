from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Verdict(str, Enum):
    """Standardized verdict values used by public GEMLA result objects."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class TrajectoryInput:
    """Validated real-valued trajectory input."""

    values: np.ndarray
    name: str = "trajectory"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("TrajectoryInput.values must be a 2D array.")
        if values.shape[0] < 3:
            raise ValueError("TrajectoryInput.values must contain at least 3 time steps.")
        if not np.all(np.isfinite(values)):
            raise ValueError("TrajectoryInput.values must contain only finite values.")
        object.__setattr__(self, "values", values)

    @property
    def n_steps(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True)
class ComplexTrajectory:
    """Validated one-dimensional complex trajectory."""

    values: np.ndarray
    name: str = "complex_trajectory"

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=complex)
        if values.ndim != 1:
            raise ValueError("ComplexTrajectory.values must be a 1D array.")
        if values.shape[0] < 3:
            raise ValueError("ComplexTrajectory.values must contain at least 3 time steps.")
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ValueError("ComplexTrajectory.values must contain only finite values.")
        object.__setattr__(self, "values", values)

    @property
    def n_steps(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class GateDecision:
    """Typed public contract for a single diagnostic gate decision."""

    name: str
    passed: bool
    score: float | None = None
    threshold: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def verdict(self) -> Verdict:
        return Verdict.PASS if self.passed else Verdict.FAIL

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


@dataclass(frozen=True)
class ControlDecision:
    """Typed public contract for a control branch decision."""

    name: str
    rejected: bool
    passed_gate: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def verdict(self) -> Verdict:
        return Verdict.PASS if self.rejected else Verdict.FAIL

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


@dataclass(frozen=True)
class AuditArtifact:
    """Small serializable record describing an output artifact."""

    name: str
    path: str
    kind: str
    description: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)