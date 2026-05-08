import numpy as np


def make_controls(z: np.ndarray, seed: int = 11) -> dict[str, np.ndarray]:
    """
    Generate simple adversarial controls for a complex trajectory.
    """
    rng = np.random.default_rng(seed)
    z = np.asarray(z)

    phase = np.angle(z)
    amp = np.abs(z)

    shuffled_idx = rng.permutation(len(z))

    wrong_sign = np.conjugate(z)
    phase_shuffle = amp * np.exp(1j * phase[shuffled_idx])

    residue_scramble = z.copy()
    residue_scramble = residue_scramble[shuffled_idx]

    return {
        "wrong_sign": wrong_sign,
        "phase_shuffle": phase_shuffle,
        "residue_scramble": residue_scramble,
    }