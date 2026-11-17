"""Domain exceptions raised by the dataset tooling."""


class FootDataError(RuntimeError):
    """Base error for an actionable dataset problem."""


class UnsafeArchiveError(FootDataError):
    """An archive failed path, size, encryption, or compression checks."""


class ManifestError(FootDataError):
    """A manifest is malformed or internally inconsistent."""


class SplitError(FootDataError):
    """A leakage-safe split cannot be produced."""


