import numpy as np

from gemla.data import make_synthetic_transport
from gemla.lifted import complex_from_xy, lifted_phase, winding_summary


def test_lifted_phase_is_finite():
    X = make_synthetic_transport()
    z = complex_from_xy(X)
    theta = lifted_phase(z)

    assert theta.ndim == 1
    assert len(theta) == len(X)
    assert np.all(np.isfinite(theta))


def test_winding_has_jumps():
    X = make_synthetic_transport()
    z = complex_from_xy(X)
    theta = lifted_phase(z)

    winding = winding_summary(theta)

    assert winding["jump_count"] > 0
    assert winding["total_winding_cells"] > 0