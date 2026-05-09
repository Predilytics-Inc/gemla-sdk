# Release Checklist

Before creating a release: 

## Local checks 
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

# Git Checks
```bash
git status
```

Confirm no generated reports, .npy files, cache files, or benchmark outputs are staged.

# Required Files
- README.md
- LICENSE
- CITATION.cff
- CHANGELOG.md
- docs/architecture_card.md
- docs/quickstart.md
- docs/cli_reference.md
- docs/limitations.md
- docs/examples.md

# GitHub Checks
- GitHub Actions passing
- PR reviewed
- Release tag created
- Release notes added

