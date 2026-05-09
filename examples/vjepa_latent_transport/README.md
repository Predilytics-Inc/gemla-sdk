# V-JEPA-Style Latent Transport Demo

This example shows how GEMLA can evaluate externally extracted video/world-model embeddings.

The demo does not download or redistribute V-JEPA weights. Instead, it uses synthetic V-JEPA-like embeddings to demonstrate the integration path:

```text
video/world-model embeddings
→ GEMLA latent adapter
→ complex surrogate trajectory
→ lifted phase
→ RevA v2-SF gate
→ adversarial controls
→ anchor and winding diagnostics
→ transport verdict
```

## Run

```bash
python examples/vjepa_latent_transport/run_vjepa_latent_transport_demo.py
```

## Using real embeddings later

Real external embeddings should be saved as a NumPy `.npy` file.

Expected shape:

```text
(n_steps, latent_dim)
```

Example:

```python
from gemla.integrations.vjepa import load_vjepa_embeddings
from gemla.pipelines import GemlaLatentPipeline

embeddings = load_vjepa_embeddings("path/to/embeddings.npy")
result = GemlaLatentPipeline().fit_evaluate(embeddings)

print(result.summary())
```

This adapter treats V-JEPA-style embeddings as a surrogate transport trajectory. GEMLA does not claim ownership of or redistribute third-party model weights.