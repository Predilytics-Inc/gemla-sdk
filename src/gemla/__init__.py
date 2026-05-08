from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gemla")
except PackageNotFoundError:
    # Fallback when running directly from source before installation.
    __version__ = "0.1.0"

__all__ = ["__version__"]