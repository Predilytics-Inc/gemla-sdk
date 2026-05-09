# V-JEPA-Style Latent Transport Demo

This example shows how GEMLA can evaluate externally extracted video/world-model embeddings.

The demo does **not** download or redistribute V-JEPA weights. Instead, it uses synthetic V-JEPA-like embeddings to demonstrate the integration path:

```text
video/world-model embeddings
→ GEMLA latent adapter
→ complex surrogate trajectory
→ lifted phase
→ RevA v2-SF gate
→ adversarial controls
→ anchor and winding diagnostics
→ transport verdict