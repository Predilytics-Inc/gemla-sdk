# Package Build

## Build

```bash
python -m pip install --upgrade build
python -m build
```

This creates:

```text
dist/
  gemla-<version>-py3-none-any.whl
  gemla-<version>.tar.gz
```

## Test wheel install

```bash
python -m venv .venv-wheel-test
.venv-wheel-test\Scripts\Activate.ps1
pip install dist/gemla-<version>-py3-none-any.whl
gemla --help
gemla evaluate
deactivate
```

## Notes

Generated build outputs are ignored by Git:

```text
dist/
build/
*.egg-info/
```
