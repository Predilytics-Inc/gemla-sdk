from __future__ import annotations

from dataclasses import dataclass, field

from gemla.exceptions import GemlaConfigurationError


@dataclass(frozen=True)
class GemlaConfig:
    """Runtime configuration shared by GEMLA 1.x and future GEMLA 2.x pipelines.

    Version 1.1.0 keeps this intentionally small. Later releases will extend this
    object with ADL, calibration, cross-resolution, and T4f settings without
    changing the public configuration pattern.
    """

    seed: int = 11
    eps: float = 1e-12
    strict: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.seed, int):
            raise GemlaConfigurationError("seed must be an integer.")
        if self.eps <= 0:
            raise GemlaConfigurationError("eps must be positive.")
        if not isinstance(self.strict, bool):
            raise GemlaConfigurationError("strict must be a boolean.")
        if not isinstance(self.metadata, dict):
            raise GemlaConfigurationError("metadata must be a dictionary.")

    @classmethod
    def default(cls) -> "GemlaConfig":
        config = cls()
        config.validate()
        return config