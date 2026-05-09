# Install 
From the repository root: 

```bash
pip install -e ".[dev]"
```

# Check Install
```bash
gemla --help
```
# Run the default Evaluation 
```bash
gemla evaluate
```
# Run Benchmarks
```bash
gemla benchmark
```
# Run embedding evaluation
```bash
gemla evaluate-latent
```
# Run V-JEPA-style evaluation 
Synthetic Demo:

```bash
gemla evaluate-vjepa --synthetic
```
External Embeddings: 
```bash
gemla evaluate-vjepa --input path/to/embeddings.npy
```
Expected embedding shape: 
(n_steps, latent_dim)

# Run practical demos
```bash
gemla demo-industrial
gemla demo-market
gemla demo-cyber
```
# Run tests
```bash
pytest
```
