## 💖 Support This Project

If this framework is useful or relevant to your work, you can support its continued development:

[![Sponsor](https://img.shields.io/badge/Sponsor-GEMLA-red)](https://github.com/sponsors/Predilytics-Inc)

👉 https://github.com/sponsors/Predilytics-Inc

Sponsorship helps fund:
- core SDK development
- research and experimental validation
- benchmark infrastructure
- documentation and reproducibility tooling

# GEMLA SDK

[![PyPI version](https://img.shields.io/pypi/v/gemla.svg)](https://pypi.org/project/gemla/)
[![Python versions](https://img.shields.io/pypi/pyversions/gemla.svg)](https://pypi.org/project/gemla/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20073321.svg)](https://doi.org/10.5281/zenodo.20073321)

**GEMLA** is an open-source Python SDK for lifted-phase transport diagnostics across trajectories, telemetry, latent embeddings, and multi-signal systems.

GEMLA evaluates whether a supplied trajectory produces a stable lifted transport signature that passes diagnostic gates and rejects adversarial controls.

At a high level:

```text
trajectory / embeddings
→ lifted phase construction
→ RevA v2-SF gate
→ adversarial controls
→ anchor diagnostics
→ winding diagnostics
→ PASS / FAIL verdict
```

GEMLA can be used with synthetic trajectories, industrial telemetry, market microstructure proxies, cyber event streams, latent embeddings, and V-JEPA-style embedding arrays.

---

## Install

Install from PyPI:

```bash
pip install gemla==1.0.0
```

Or install the latest available release:

```bash
pip install gemla
```

Check the installation:

```bash
gemla --help
```

Run the default demo:

```bash
gemla evaluate
```

---

## Paper and Experimental Archive

The accompanying paper and corresponding experimental results are available here:

https://doi.org/10.5281/zenodo.20073321

The archive contains the research paper, experimental outputs, validation materials, and supporting results related to the GEMLA architecture and transport diagnostics framework.

---

## Current Status

**Current release:** `v1.0.0`

Current capabilities:

- PyPI package install
- command-line interface
- synthetic transport generation
- lifted phase construction
- RevA v2-SF gate evaluation
- wrong-sign, phase-shuffle, and residue-scramble controls
- anchor irregularity diagnostics
- integer winding diagnostics
- Markdown report export
- benchmark suite
- latent embedding evaluation
- V-JEPA-style embedding adapter
- industrial telemetry demo
- market microstructure demo
- cyber event transport demo
- pytest validation
- GitHub Actions CI
- GitHub release artifacts

---

## Why GEMLA Exists

Many complex systems do not fail or change through a single obvious spike. Their important signals may appear through how the system moves through states over time.

GEMLA provides a reusable diagnostic layer for these evolving systems. It focuses on trajectory structure, lifted-phase organization, adversarial control rejection, anchor behavior, and winding behavior.

The goal is to help researchers and builders test whether a trajectory contains coherent transport structure rather than only isolated amplitude changes or noise artifacts.

---

## Where GEMLA Can Be Used

GEMLA is designed as a general trajectory-diagnostics SDK. It can be applied anywhere a system produces time-varying signals, telemetry, embeddings, or event streams.

| Area | Example input | Possible diagnostic use |
|---|---|---|
| Industrial systems | temperature, pressure, vibration, machine usage | process drift and early instability |
| Finance / markets | price, volume, volatility, order-flow proxies | regime transition diagnostics |
| Cybersecurity | login events, alerts, access patterns | coordinated abnormal event progression |
| AI monitoring | embeddings, traces, latent states, tool-use sequences | behavioral mode shifts |
| World models | video embeddings or V-JEPA-style latent arrays | latent trajectory changes |
| Healthcare research | wearable signals, sensor streams, patient time-series | physiological regime-shift research |
| Energy systems | grid telemetry, load curves, equipment sensors | operating-state instability |
| Logistics / operations | route timing, demand signals, fleet telemetry | disruption pattern detection |
| Scientific data | multi-signal experimental measurements | weak structure discovery |

GEMLA is not tied to one industry dataset. Once data is represented as a trajectory or embedding stream, it can be passed through the same diagnostic pipeline.

---

## Core Pipeline

```text
source trajectory
→ complex / latent surrogate
→ lifted phase Θ(t)
→ RevA v2-SF gate
→ adversarial controls
→ anchor diagnostics
→ winding diagnostics
→ PASS / FAIL verdict
```

The public SDK includes both a default synthetic trajectory pipeline and domain-style demos that use the same gate stack.

---

## Quickstart

Run the minimal vertical-slice pipeline:

```bash
gemla evaluate
```

Expected output includes:

```text
GEMLA Transport Evaluation
--------------------------
RevA v2-SF main pass: True
Wrong-sign rejected: True
Phase-shuffle rejected: True
Residue-scramble rejected: True
Final verdict: PASS
```

The command writes a Markdown report to:

```text
reports/gemla_transport_report.md
```

---

## CLI Usage

### Default transport evaluation

```bash
gemla evaluate
```

Custom output:

```bash
gemla evaluate --output reports/custom_report.md
```

Custom synthetic trajectory:

```bash
gemla evaluate --n 1500 --t-max 100 --noise 0.02 --seed 9
```

### Benchmark suite

```bash
gemla benchmark
```

Custom benchmark output directory:

```bash
gemla benchmark --output-dir benchmarks/results
```

### Latent embedding evaluation

Run synthetic latent evaluation:

```bash
gemla evaluate-latent
```

Evaluate saved latent embeddings:

```bash
gemla evaluate-latent --input path/to/latents.npy
```

Expected input shape:

```text
(n_steps, latent_dim)
```

### V-JEPA-style embedding evaluation

Run synthetic V-JEPA-style demo:

```bash
gemla evaluate-vjepa --synthetic
```

Evaluate external embeddings:

```bash
gemla evaluate-vjepa --input path/to/embeddings.npy
```

GEMLA does not download or redistribute third-party model weights. The V-JEPA-style adapter treats external embeddings as surrogate transport trajectories.

### Practical deployment demos

Industrial telemetry:

```bash
gemla demo-industrial
```

Market microstructure:

```bash
gemla demo-market
```

Cyber event transport:

```bash
gemla demo-cyber
```

Each command generates synthetic domain-proxy data, runs the GEMLA lifted-phase gate stack, rejects adversarial controls, and writes a Markdown report.

---

## Python Examples

Minimal transport example:

```bash
python examples/quickstart/01_minimal_pipeline.py
```

Industrial telemetry:

```bash
python examples/industrial_telemetry/run_industrial_telemetry_demo.py
```

Market microstructure:

```bash
python examples/market_microstructure/run_market_microstructure_demo.py
```

Cyber event transport:

```bash
python examples/cybersecurity/run_cyber_transport_demo.py
```

Latent embeddings:

```bash
python examples/latent_embeddings/run_latent_transport_demo.py
```

V-JEPA-style embeddings:

```bash
python examples/vjepa_latent_transport/run_vjepa_latent_transport_demo.py
```

---

## What a PASS Means

A `PASS` means the supplied trajectory satisfied the current GEMLA diagnostic gate stack and rejected the included adversarial controls.

A `PASS` does **not** mean the system is safe, optimal, causal in a physical sense, production-ready, or suitable for high-stakes deployment without domain validation.

GEMLA v1.0.0 is a research SDK. It provides reproducible diagnostics, synthetic/domain-proxy demos, benchmark workflows, and example adapters.

---

## Development Setup

For contributors who want to work from source:

```bash
git clone https://github.com/Predilytics-Inc/gemla-sdk.git
cd gemla-sdk
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Build package artifacts:

```bash
python -m build
```

---

## Repository Structure

```text
src/gemla/
  data/          synthetic and domain-proxy data generators
  lifted/        lifted phase, anchors, spectral flatness, winding
  gates/         RevA v2-SF gate evaluation
  controls/      adversarial control generation
  pipelines/     end-to-end GEMLA pipelines
  integrations/  latent and V-JEPA-style adapters
  benchmarks/    benchmark runner and result writers
  reports/       Markdown report exporter
  cli/           command-line interface

examples/
  quickstart/
  industrial_telemetry/
  market_microstructure/
  cybersecurity/
  latent_embeddings/
  vjepa_latent_transport/

tests/
  unit and integration tests
```

---

## Documentation

- [Architecture Card](docs/architecture_card.md)
- [Quickstart](docs/quickstart.md)
- [CLI Reference](docs/cli_reference.md)
- [Examples](docs/examples.md)
- [Limitations](docs/limitations.md)
- [Release Checklist](docs/release_checklist.md)
- [Changelog](CHANGELOG.md)

---

## Contributing

Contributions are welcome.

Useful ways to contribute:

- open issues for bugs or unclear documentation
- improve examples
- add real-data adapter examples
- add benchmark cases
- improve tests
- propose clearer terminology
- submit pull requests

Before using GEMLA in a domain-specific or high-stakes setting, validate the pipeline against domain-specific data, controls, and expert review.

---

## Support GEMLA

GEMLA is an open-source research SDK developed to make lifted-phase transport diagnostics reproducible and accessible.

If GEMLA is useful to your research, engineering work, or experiments, you can support continued development by:

- starring the repository
- citing the project
- sharing feedback
- opening issues
- contributing pull requests
- sponsoring development through the repository funding links

Sponsorship helps fund core SDK development, research validation, benchmark infrastructure, documentation, and reproducibility tooling.

---

## Trademark Notice

GEMLA, Hybrid Γ–EML–α, and related branding are trademarks of Predilytics Inc.

---

## License

See [LICENSE](LICENSE).

---

## Citation

See [CITATION.cff](CITATION.cff).
