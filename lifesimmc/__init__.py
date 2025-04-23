try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version  # For Python <3.8

try:
    __version__ = version("lifesimmc")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
