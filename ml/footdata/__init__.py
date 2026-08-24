"""Reproducible, leakage-aware tooling for foot segmentation datasets."""

__version__ = "0.1.0"

from .errors import FootDataError, ManifestError, SplitError, UnsafeArchiveError
from .importers import ImportResult, import_dataset
from .manifest import read_manifest, validate_manifest, write_manifest
from .split import split_records

__all__ = [
    "FootDataError",
    "ImportResult",
    "ManifestError",
    "SplitError",
    "UnsafeArchiveError",
    "import_dataset",
    "read_manifest",
    "split_records",
    "validate_manifest",
    "write_manifest",
]
