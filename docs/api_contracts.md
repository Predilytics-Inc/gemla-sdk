# GEMLA 1.1.0 API Contracts

GEMLA 1.1.0 introduces small, stable public contracts that later releases can extend without breaking existing users.

## Configuration

```python
from gemla import GemlaConfig

config = GemlaConfig.default()
```

## Trajectory input

```python
from gemla import TrajectoryInput

trajectory = TrajectoryInput(X, name="demo")
```

A `TrajectoryInput` requires a finite two-dimensional array with at least three time steps.

## Complex trajectory

```python
from gemla import ComplexTrajectory

z = ComplexTrajectory(values)
```

A `ComplexTrajectory` requires a finite one-dimensional complex array with at least three time steps.

## Gate decision

```python
from gemla import GateDecision

decision = GateDecision(
    name="spectral_flatness",
    passed=True,
    score=0.42,
    threshold=1.25,
)
```

## Control decision

```python
from gemla import ControlDecision

control = ControlDecision(
    name="wrong_sign",
    rejected=True,
    passed_gate=False,
)
```

## Audit artifact

```python
from gemla import AuditArtifact

artifact = AuditArtifact(
    name="report",
    path="reports/gemla_report.md",
    kind="markdown",
)
```

Audit artifacts are introduced in 1.1.0 so that 1.9.0 and 2.0.0 can export CSV/JSON/ZIP evidence bundles using the same pattern.
