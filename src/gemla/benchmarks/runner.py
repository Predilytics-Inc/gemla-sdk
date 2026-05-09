from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gemla.data import make_synthetic_transport
from gemla.pipelines import GemlaPipeline


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    n: int = 1200
    t_max: float = 80.0
    noise: float = 0.01
    seed: int = 7
    pipeline_seed: int = 11


@dataclass(frozen=True)
class BenchmarkRecord:
    name: str
    verdict: bool
    spectral_flatness_score: float
    wrong_sign_rejected: bool
    phase_shuffle_rejected: bool
    residue_scramble_rejected: bool
    anchor_count: int
    anchor_spacing_cv: float
    winding_jumps: int
    winding_cells: int
    n: int
    t_max: float
    noise: float
    seed: int
    pipeline_seed: int


def default_benchmark_cases() -> list[BenchmarkCase]:
    """
    Minimal v0.2 benchmark suite.

    These cases test whether the vertical slice remains stable across
    modest changes in seed, noise, and trajectory length.
    """
    return [
        BenchmarkCase(name="baseline", n=1200, t_max=80.0, noise=0.01, seed=7),
        BenchmarkCase(name="low_noise", n=1200, t_max=80.0, noise=0.005, seed=7),
        BenchmarkCase(name="no_noise", n=1200, t_max=80.0, noise=0.0, seed=7),
        BenchmarkCase(name="alt_seed_1", n=1200, t_max=80.0, noise=0.01, seed=13),
        BenchmarkCase(name="alt_seed_2", n=1200, t_max=80.0, noise=0.01, seed=29),
        BenchmarkCase(name="shorter_series", n=800, t_max=60.0, noise=0.01, seed=7),
        BenchmarkCase(name="longer_series", n=1600, t_max=100.0, noise=0.01, seed=7),
    ]


def run_benchmark_case(case: BenchmarkCase) -> BenchmarkRecord:
    X = make_synthetic_transport(
        n=case.n,
        t_max=case.t_max,
        noise=case.noise,
        seed=case.seed,
    )

    result = GemlaPipeline(seed=case.pipeline_seed).fit_evaluate(X)

    main = result.gate_result["main"]
    controls = result.gate_result["controls"]

    return BenchmarkRecord(
        name=case.name,
        verdict=result.verdict,
        spectral_flatness_score=float(main["flatness"]["score"]),
        wrong_sign_rejected=bool(controls["wrong_sign"]["rejected"]),
        phase_shuffle_rejected=bool(controls["phase_shuffle"]["rejected"]),
        residue_scramble_rejected=bool(controls["residue_scramble"]["rejected"]),
        anchor_count=int(result.anchor_result["anchor_count"]),
        anchor_spacing_cv=float(result.anchor_result["spacing_cv"]),
        winding_jumps=int(result.winding_result["jump_count"]),
        winding_cells=int(result.winding_result["total_winding_cells"]),
        n=case.n,
        t_max=case.t_max,
        noise=case.noise,
        seed=case.seed,
        pipeline_seed=case.pipeline_seed,
    )


def run_benchmark_suite(
    cases: list[BenchmarkCase] | None = None,
) -> list[BenchmarkRecord]:
    if cases is None:
        cases = default_benchmark_cases()

    return [run_benchmark_case(case) for case in cases]


def summarize_benchmark_records(records: list[BenchmarkRecord]) -> dict:
    total = len(records)
    passed = sum(record.verdict for record in records)

    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "all_passed": passed == total,
        "all_wrong_sign_rejected": all(r.wrong_sign_rejected for r in records),
        "all_phase_shuffle_rejected": all(r.phase_shuffle_rejected for r in records),
        "all_residue_scramble_rejected": all(r.residue_scramble_rejected for r in records),
        "min_anchor_count": min((r.anchor_count for r in records), default=0),
        "min_winding_jumps": min((r.winding_jumps for r in records), default=0),
    }


def write_benchmark_results(
    records: list[BenchmarkRecord],
    output_dir: str | Path = "benchmarks/results",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_results.json"
    summary_path = output_dir / "benchmark_summary.json"

    rows = [asdict(record) for record in records]

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary = summarize_benchmark_records(records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "csv": csv_path,
        "json": json_path,
        "summary": summary_path,
    }