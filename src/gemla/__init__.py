from __future__ import annotations

try:
    from gemla.__version import __version__
except Exception:  # pragma: no cover
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            __version__ = version("gemla")
        except PackageNotFoundError:
            __version__ = "1.1.0"
    except Exception:
        __version__ = "1.1.0"

from gemla.config import GemlaConfig
from gemla.exceptions import (
    GemlaConfigurationError,
    GemlaError,
    GemlaInputError,
    GemlaValidationError,
)
from gemla.types import (
    AuditArtifact,
    ComplexTrajectory,
    ControlDecision,
    GateDecision,
    TrajectoryInput,
    Verdict,
)

__all__ = [
    "__version__",
    "GemlaConfig",
    "GemlaError",
    "GemlaInputError",
    "GemlaValidationError",
    "GemlaConfigurationError",
    "TrajectoryInput",
    "ComplexTrajectory",
    "GateDecision",
    "ControlDecision",
    "AuditArtifact",
    "Verdict",
]