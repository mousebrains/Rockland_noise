"""Compare spikes and noise between two channels across Rockland P-files."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rockland-noise")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "unknown"
