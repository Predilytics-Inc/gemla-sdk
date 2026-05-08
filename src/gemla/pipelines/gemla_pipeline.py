from dataclasses import dataclass

import numpy as np

from gemla.controls import make_controls
from gemla.gates import evaluate_reva_v2_sf
from gemla.lifted import anchor_summary, complex_from_xy, lifted_phase, winding_summary


@dataclass
class GemlaResult:
    verdict: bool
    gate_result: dict
    anchor_result: dict
    winding_result: dict

    def summary(self) -> str:
        main = self.gate_result["main"]
        controls = self.gate_result["controls"]

        lines = [
            "GEMLA Transport Evaluation",
            "--------------------------",
            f"RevA v2-SF main pass: {main['pass']}",
            f"Spectral flatness score: {main['flatness']['score']:.4f}",
            f"Wrong-sign rejected: {controls['wrong_sign']['rejected']}",
            f"Phase-shuffle rejected: {controls['phase_shuffle']['rejected']}",
            f"Residue-scramble rejected: {controls['residue_scramble']['rejected']}",
            f"Anchor count: {self.anchor_result['anchor_count']}",
            f"Anchor spacing CV: {self.anchor_result['spacing_cv']:.4f}",
            f"Winding jumps: {self.winding_result['jump_count']}",
            f"Winding cells: {self.winding_result['total_winding_cells']}",
            f"Final verdict: {'PASS' if self.verdict else 'FAIL'}",
        ]

        return "\n".join(lines)


class GemlaPipeline:
    """
    Minimal GEMLA v0.1 vertical slice.

    Input:
        X(t)

    Processing:
        X -> complex surrogate z(t)
        z -> lifted phase Θ(t)
        Θ -> RevA v2-SF, anchors, winding
        z -> adversarial controls

    Output:
        PASS / FAIL transport verdict
    """

    def __init__(self, seed: int = 11):
        self.seed = seed

    def fit_evaluate(self, X: np.ndarray) -> GemlaResult:
        z = complex_from_xy(X)
        theta = lifted_phase(z)

        controls = make_controls(z, seed=self.seed)
        gate_result = evaluate_reva_v2_sf(z, controls)

        anchor_result = anchor_summary(theta)
        winding_result = winding_summary(theta)

        anchor_ok = (
            anchor_result["anchor_count"] >= 3
            and anchor_result["spacing_cv"] > 0.01
        )

        winding_ok = winding_result["jump_count"] > 0

        verdict = bool(
            gate_result["pass"]
            and anchor_ok
            and winding_ok
        )

        return GemlaResult(
            verdict=verdict,
            gate_result=gate_result,
            anchor_result=anchor_result,
            winding_result=winding_result,
        )