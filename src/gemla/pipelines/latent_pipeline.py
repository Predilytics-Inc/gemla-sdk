from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gemla.controls import make_controls
from gemla.gates import evaluate_reva_v2_sf
from gemla.integrations.latent import latent_to_complex_surrogate
from gemla.lifted import anchor_summary, lifted_phase, winding_summary


@dataclass
class GemlaLatentResult:
    verdict: bool
    input_shape: tuple[int, int]
    gate_result: dict
    anchor_result: dict
    winding_result: dict

    def summary(self) -> str:
        main = self.gate_result["main"]
        controls = self.gate_result["controls"]

        lines = [
            "GEMLA Latent Transport Evaluation",
            "---------------------------------",
            f"Input shape: {self.input_shape}",
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


class GemlaLatentPipeline:
    """
    GEMLA v0.3 latent embedding vertical slice.

    Input:
        latent embedding trajectory, shape (n_steps, latent_dim)

    Processing:
        latents -> complex surrogate z(t)
        z -> lifted phase Θ(t)
        Θ -> RevA v2-SF, anchors, winding
        z -> adversarial controls

    Output:
        PASS / FAIL latent transport verdict
    """

    def __init__(
        self,
        seed: int = 11,
        component_a: int = 0,
        component_b: int = 1,
    ):
        self.seed = seed
        self.component_a = component_a
        self.component_b = component_b

    def fit_evaluate(self, latents: np.ndarray) -> GemlaLatentResult:
        latents = np.asarray(latents, dtype=float)

        z = latent_to_complex_surrogate(
            latents,
            component_a=self.component_a,
            component_b=self.component_b,
        )

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

        return GemlaLatentResult(
            verdict=verdict,
            input_shape=tuple(latents.shape),
            gate_result=gate_result,
            anchor_result=anchor_result,
            winding_result=winding_result,
        )