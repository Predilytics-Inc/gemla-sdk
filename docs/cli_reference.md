# CLI Reference

## `gemla evaluate`
Runs the minimal GEMLA synthetic transport pipeline.

```bash
gemla evaluate
```
Options: 
```bash
gemla evaluate --output reports/custom_report.md
gemla evaluate --n 1500 --t-max 100 --noise 0.02 --seed 9
```
## `gemla benchmark`
Runs the benchmark suite.

```bash
gemla benchmark
```
Options: 
```bash
gemla benchmark --output-dir benchmarks/results
```

# `gemla evaluate-latent`
Runs GEMLA on latent embedding trajectories.
```bash
gemla evaluate-latent
```
Options: 
```bash
gemla evaluate-latent --input path/to/latents.npy
gemla evaluate-latent --n 800 --latent-dim 12 --noise 0.02
```

# `gemla evaluate-vjepa`
Runs GEMLA on V-JEPA-style external embeddings.

synthetic mode:
```bash
gemla evaluate-vjepa --synthetic
```
External embedding mode: 
```bash
gemla evaluate-vjepa --input path/to/embeddings.npy
```

# `gemla demo-industrial`
Runs the industrial telemetry proxy demo.
```bash
gemla demo-industrial
```
# `gemla demo-market`
Runs the market microstructure proxy demo.
```bash 
gemla demo-market
```

# `gemla demo-cyber`
Runs the cyber event transport proxy demo.
```bash
gemla demo-cyber
```


