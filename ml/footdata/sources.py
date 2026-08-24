"""Pinned metadata and acquisition policy for supported datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .errors import FootDataError


@dataclass(frozen=True)
class SourceSpec:
    key: str
    source_id: str
    title: str
    creators: tuple[str, ...]
    version: str
    task: str
    synthetic: bool
    license_id: str
    license_url: str
    landing_url: str
    acquisition: str
    group_policy: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCES: dict[str, SourceSpec] = {
    "footnet": SourceSpec(
        key="footnet",
        source_id="footnet-zenodo-20457253",
        title="FootNet smartphone foot segmentation dataset",
        creators=("Adith Vijay", "Akash Prabhune", "Vinay R Srihari", "Akhila Rayampalli"),
        version="zenodo-record-20457253",
        task="binary-foot-segmentation",
        synthetic=False,
        license_id="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        landing_url="https://zenodo.org/records/20457253",
        acquisition="local-archive-or-directory",
        group_policy="subject-id; unresolved subjects are train-only",
        notes=(
            "The public release contains 191 images. Pin record 20457253 and "
            "record the local archive checksum rather than following latest."
        ),
    ),
    "synfoot-v1": SourceSpec(
        key="synfoot-v1",
        source_id="synfoot-v1",
        title="SynFoot V1 synthetic foot dataset",
        creators=("Oliver Boyne", "Gwangbin Bae", "James Charles", "Roberto Cipolla"),
        version="v1",
        task="binary-foot-segmentation",
        synthetic=True,
        license_id="LicenseRef-SynFoot-Archive-Terms",
        license_url="https://github.com/OllieBoyne/SynFoot",
        landing_url="https://github.com/OllieBoyne/SynFoot",
        acquisition="manual-google-form-then-local-import",
        group_policy="labels.json foot.name (underlying Foot3D scan)",
        notes=(
            "Download manually through the publisher form. Confirm and retain "
            "the terms bundled with the archive; repository licensing alone is "
            "not treated as proof of dataset licensing."
        ),
    ),
    "synfoot-v2": SourceSpec(
        key="synfoot-v2",
        source_id="synfoot-v2",
        title="SynFoot V2 synthetic foot dataset",
        creators=("Oliver Boyne", "Roberto Cipolla"),
        version="v2",
        task="binary-foot-segmentation",
        synthetic=True,
        license_id="LicenseRef-SynFoot-Archive-Terms",
        license_url="https://github.com/OllieBoyne/SynFoot",
        landing_url="https://github.com/OllieBoyne/SynFoot",
        acquisition="manual-google-form-then-local-import",
        group_policy="labels.json foot-id (underlying Foot3D scan)",
        notes=(
            "Download manually through the publisher form. Confirm and retain "
            "the terms bundled with the archive."
        ),
    ),
}


def get_source(key: str) -> SourceSpec:
    try:
        return SOURCES[key]
    except KeyError as error:
        choices = ", ".join(sorted(SOURCES))
        raise FootDataError(f"Unsupported source {key!r}; choose one of: {choices}") from error


def list_sources() -> list[dict[str, object]]:
    return [SOURCES[key].to_dict() for key in sorted(SOURCES)]
