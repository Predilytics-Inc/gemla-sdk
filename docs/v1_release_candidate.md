# GEMLA v1.0 Release Candidate Checklist

## Required Validation

```bash
pip install -e ".[dev]"
pytest
gemla --help
gemla evaluate
gemla benchmark
gemla evaluate-latent
gemla evaluate-vjepa --synthetic
gemla demo-industrial
gemla demo-market
gemla demo-cyber
```

# Package Validation 
```bash
python -m build
python -m twine check dist/*
```

# TestPyPi
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple gemla==1.0.0rc1
gemla --help
gemla evaluate
```

## Release Decision
Promote to 1.0.0 only after:

- GitHub Actions pass
- TestPyPI install works
- GitHub Release artifacts attach correctly
- README and docs are current
- no generated files are tracked
- no secrets are present


