# GEMLA SDK

**GEMLA** is a Γ–EML–α transport architecture research SDK for observable path measures, lifted-phase diagnostics, adversarial-control gates, and finite transport-topology evaluation.

The SDK evaluates whether a supplied trajectory produces a stable lifted transport signature that passes diagnostic gates and rejects adversarial controls.

---

## Current Status

**Research preview:** `v0.5.0-practical-demos`

Current capabilities:

- package import and editable install
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

---

## Install

From the repository root:

```bash
pip install -e ".[dev]"
```

Check installation:

```bash
gemla --help
```

---

## Quickstart

Run the minimal vertical-slice pipeline:

```bash
gemla evaluate
```

Or run the original Python example:

```bash
python examples/quickstart/01_minimal_pipeline.py
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

## Run Tests

```bash
pytest
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

## Interpretation

The current version is a finite diagnostic SDK. It does not make universal claims. A `PASS` means the supplied trajectory satisfied the current GEMLA gate stack and rejected the included adversarial controls. A `PASS` does not mean the system is safe, optimal, causal in a physical sense, or ready for high-stakes deployment.

---

## Roadmap

Planned next steps:

- documentation site
- sample output reports
- richer benchmark summaries
- configurable gate thresholds
- real-data adapter examples
- hosted demo or release bundle
- public release preparation

---

## License

See [LICENSE](LICENSE).

---

## Citation

See [CITATION.cff](CITATION.cff).