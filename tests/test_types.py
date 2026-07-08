import numpy as np
import pytest

from gemla import (
    AuditArtifact,
    ComplexTrajectory,
    ControlDecision,
    GateDecision,
    GemlaConfig,
    TrajectoryInput,
    Verdict,
)
from gemla.exceptions import GemlaConfigurationError


def test_default_config_validates():
    config = GemlaConfig.default()
    assert config.seed == 11
    assert config.eps > 0


def test_invalid_config_rejected():
    config = GemlaConfig(eps=0)
    with pytest.raises(GemlaConfigurationError):
        config.validate()


def test_trajectory_input_contract():
    X = np.ones((5, 2))
    trajectory = TrajectoryInput(X, name="demo")
    assert trajectory.n_steps == 5
    assert trajectory.n_features == 2


def test_trajectory_input_rejects_bad_shape():
    with pytest.raises(ValueError):
        TrajectoryInput(np.ones(5))


def test_complex_trajectory_contract():
    z = ComplexTrajectory(np.array([1 + 1j, 2 + 0j, 3 - 1j]))
    assert z.n_steps == 3


def test_gate_decision_contract():
    decision = GateDecision(name="demo_gate", passed=True, score=0.1, threshold=0.2)
    assert decision.verdict == Verdict.PASS
    data = decision.to_dict()
    assert data["name"] == "demo_gate"
    assert data["verdict"] == "PASS"


def test_control_decision_contract():
    decision = ControlDecision(name="wrong_sign", rejected=True, passed_gate=False)
    assert decision.verdict == Verdict.PASS
    data = decision.to_dict()
    assert data["rejected"] is True


def test_audit_artifact_contract():
    artifact = AuditArtifact(
        name="report",
        path="reports/gemla_report.md",
        kind="markdown",
    )
    assert artifact.to_dict()["kind"] == "markdown"