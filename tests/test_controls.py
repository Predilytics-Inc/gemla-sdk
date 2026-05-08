from gemla.controls import make_controls
from gemla.data import make_synthetic_transport
from gemla.lifted import complex_from_xy


def test_controls_are_created():
    X = make_synthetic_transport()
    z = complex_from_xy(X)

    controls = make_controls(z)

    assert "wrong_sign" in controls
    assert "phase_shuffle" in controls
    assert "residue_scramble" in controls

    assert controls["wrong_sign"].shape == z.shape
    assert controls["phase_shuffle"].shape == z.shape
    assert controls["residue_scramble"].shape == z.shape