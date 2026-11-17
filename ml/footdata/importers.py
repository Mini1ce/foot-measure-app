"""Local FootNet and SynFoot importers with normalized, traceable assets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__
from .archive import ArchiveLimits, safe_extract_zip
from .errors import FootDataError
from .manifest import sha256_file, write_json, write_manifest
from .qa import analyze_mask, normalize_mask
from .sources import SourceSpec, get_source


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
# Bound decoded allocations independently of compressed archive limits.  The
# largest image in the pinned FootNet release is about 16 MP (3468 x 4624), so
# these limits leave substantial headroom without permitting an image header to
# trigger an unbounded Pillow allocation.
MAX_DECODED_DIMENSION = 16_384
MAX_DECODED_PIXELS = 50_000_000
IMAGE_DIRECTORY_TOKENS = {
    "color",
    "colour",
    "image",
    "images",
    "input",
    "inputs",
    "original",
    "originals",
    "photo",
    "photos",
    "rgb",
}
MASK_DIRECTORY_TOKENS = {
    "annotation",
    "annotations",
    "groundtruth",
    "ground-truth",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
    "segmentation",
    "segmentations",
}
MODALITY_TOKENS = IMAGE_DIRECTORY_TOKENS | MASK_DIRECTORY_TOKENS
MASK_STEM_PATTERN = re.compile(
    r"(?i)(?:^|[_\-.])(mask|masked|label|labels|seg|segmentation|groundtruth|gt)(?:$|[_\-.])"
)
SUBJECT_PATTERNS = (
    re.compile(r"(?i)(?:subject|subj|patient|participant|person)[_-]?(\d+[a-z]?)"),
    re.compile(r"(?i)(?:^|[_-])p(?:erson)?[_-]?(\d{2,})(?:[_-]|$)"),
)
VIEW_ALIASES = {
    "dorsal": "dorsal",
    "top": "dorsal",
    "plantar": "plantar",
    "bottom": "plantar",
    "sole": "plantar",
    "medial": "medial",
    "lateral": "lateral",
    "anterior": "anterior",
    "front": "anterior",
    "posterior": "posterior",
    "rear": "posterior",
}
# This correction is pinned to publisher content, not to dimensions alone. A
# reversed-size heuristic would silently rotate unrelated or future samples.
FOOTNET_CLOCKWISE_ROTATIONS = {
    "42d02d5dabed4c40dc45c04baf9d328d781d9b7905e3e59d9d393d016d017709": (
        "righttop_004.jpg"
    ),
}


@dataclass
class ImportResult:
    records: list[dict[str, Any]]
    manifest_path: Path
    lock_path: Path
    errors: list[str]
    warnings: list[str]

    @property
    def valid_count(self) -> int:
        return sum(bool(record["qa"]["valid"]) for record in self.records)


@dataclass(frozen=True)
class _Pair:
    key: str
    image: Path
    mask: Path


@dataclass(frozen=True)
class _SynFootV2Sample:
    key: str
    image: Path
    mask: Path
    label: Path


@dataclass(frozen=True)
class _SynFootV2Layout:
    image_directory: Path
    mask_directory: Path
    label_directory: Path


@dataclass(frozen=True)
class _SynFootV2Label:
    foot_id: str
    camera_position: tuple[float, float, float]
    camera_look_at: tuple[float, float, float]
    camera_tilt_deg: float
    camera_azimuth_deg: float
    foot_poses: dict[str, Any] | None
    foot_shape: dict[str, Any] | None


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return normalized or "unknown"


def _digest(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _clean_modality_part(value: str) -> str:
    tokens = [token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token]
    retained = [token for token in tokens if token not in MODALITY_TOKENS]
    return "-".join(retained)


def _clean_stem(stem: str) -> str:
    value = MASK_STEM_PATTERN.sub("-", stem.casefold())
    tokens = [token for token in re.split(r"[^a-z0-9]+", value) if token]
    while tokens and tokens[0] in MODALITY_TOKENS:
        tokens.pop(0)
    while tokens and tokens[-1] in MODALITY_TOKENS:
        tokens.pop()
    return "-".join(tokens) or _slug(stem)


def _canonical_key(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    directories = [_clean_modality_part(part) for part in relative.parts[:-1]]
    retained = [part for part in directories if part]
    retained.append(_clean_stem(path.stem))
    return "/".join(retained)


def _is_mask_path(path: Path, root: Path) -> bool:
    relative_parts = [part.casefold() for part in path.relative_to(root).parts[:-1]]
    for part in relative_parts:
        tokens = {token for token in re.split(r"[^a-z0-9-]+", part) if token}
        if tokens & MASK_DIRECTORY_TOKENS:
            return True
    return MASK_STEM_PATTERN.search(path.stem) is not None


def _index_unique(paths: Iterable[Path], root: Path, kind: str) -> tuple[dict[str, Path], list[str]]:
    candidates: dict[str, list[Path]] = {}
    for path in sorted(paths):
        key = _canonical_key(path, root)
        candidates.setdefault(key, []).append(path)

    index: dict[str, Path] = {}
    errors: list[str] = []
    for key, matches in candidates.items():
        if len(matches) > 1:
            errors.append(
                f"Ambiguous {kind} key {key!r}: "
                + " and ".join(path.relative_to(root).as_posix() for path in matches)
            )
            continue
        index[key] = matches[0]
    return index, errors


def discover_pairs(root: str | os.PathLike[str]) -> tuple[list[_Pair], list[str], list[str]]:
    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise FootDataError(f"Input directory does not exist: {source_root}")
    all_images = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    mask_paths = [path for path in all_images if _is_mask_path(path, source_root)]
    mask_path_set = set(mask_paths)
    image_paths = [path for path in all_images if path not in mask_path_set]
    image_index, image_errors = _index_unique(image_paths, source_root, "image")
    mask_index, mask_errors = _index_unique(mask_paths, source_root, "mask")
    errors = [*image_errors, *mask_errors]
    warnings: list[str] = []

    pairs: list[_Pair] = []
    for key in sorted(image_index.keys() & mask_index.keys()):
        pairs.append(_Pair(key=key, image=image_index[key], mask=mask_index[key]))
    for key in sorted(image_index.keys() - mask_index.keys()):
        warnings.append(
            f"Unmatched image: {image_index[key].relative_to(source_root).as_posix()}"
        )
    for key in sorted(mask_index.keys() - image_index.keys()):
        warnings.append(f"Unmatched mask: {mask_index[key].relative_to(source_root).as_posix()}")
    if not pairs:
        errors.append(
            "No image/mask pairs found. Use parallel image(s)/rgb and mask(s)/label(s) "
            "directories or a *_mask filename suffix."
        )
    return pairs, errors, warnings


def _is_ignored_publisher_path(path: Path) -> bool:
    return any(
        part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
        for part in path.parts
    )


def _select_synfoot_v2_directory(
    root: Path,
    candidates: tuple[Path, ...],
    suffix: str,
    kind: str,
) -> Path:
    matches: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            has_assets = any(
                path.is_file()
                and not _is_ignored_publisher_path(path.relative_to(root))
                and path.suffix.casefold() == suffix
                for path in candidate.iterdir()
            )
        except OSError as error:
            raise FootDataError(
                f"Cannot inspect SynFoot V2 {kind} directory {candidate}: {error}"
            ) from error
        if has_assets:
            matches.append(candidate)

    if not matches:
        expected = " or ".join(path.relative_to(root).as_posix() for path in candidates)
        raise FootDataError(
            f"Cannot find SynFoot V2 {kind} files directly under {expected}"
        )
    if len(matches) > 1:
        found = ", ".join(path.relative_to(root).as_posix() for path in matches)
        raise FootDataError(f"Ambiguous SynFoot V2 {kind} directories: {found}")
    return matches[0]


def _discover_synfoot_v2_layout(root: Path) -> _SynFootV2Layout:
    if not root.is_dir():
        raise FootDataError(f"Input directory does not exist: {root}")
    return _SynFootV2Layout(
        image_directory=_select_synfoot_v2_directory(
            root,
            (root / "rgb", root / "rgb" / "rgb"),
            ".png",
            "RGB",
        ),
        mask_directory=_select_synfoot_v2_directory(
            root,
            (root / "mask", root / "mask" / "mask"),
            ".png",
            "mask",
        ),
        label_directory=_select_synfoot_v2_directory(
            root,
            (root / "labels_processed", root / "labels" / "labels_processed"),
            ".json",
            "label",
        ),
    )


def _index_synfoot_v2_files(
    root: Path,
    directory: Path,
    suffix: str,
    kind: str,
) -> tuple[dict[str, Path], list[str]]:
    candidates: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if (
            not path.is_file()
            or _is_ignored_publisher_path(path.relative_to(root))
            or path.suffix.casefold() != suffix
        ):
            continue
        key = path.stem.casefold()
        candidates.setdefault(key, []).append(path)

    indexed: dict[str, Path] = {}
    errors: list[str] = []
    for key, matches in candidates.items():
        if len(matches) > 1:
            errors.append(
                f"Ambiguous SynFoot V2 {kind} stem {key!r}: "
                + " and ".join(path.relative_to(root).as_posix() for path in matches)
            )
            continue
        indexed[key] = matches[0]
    return indexed, errors


def _discover_synfoot_v2_samples(
    root: Path,
) -> tuple[list[_SynFootV2Sample], list[str], list[str]]:
    layout = _discover_synfoot_v2_layout(root)
    images, image_errors = _index_synfoot_v2_files(
        root, layout.image_directory, ".png", "RGB"
    )
    masks, mask_errors = _index_synfoot_v2_files(
        root, layout.mask_directory, ".png", "mask"
    )
    labels, label_errors = _index_synfoot_v2_files(
        root, layout.label_directory, ".json", "label"
    )
    errors = [*image_errors, *mask_errors, *label_errors]
    warnings: list[str] = []
    all_keys = images.keys() | masks.keys() | labels.keys()
    complete_keys = images.keys() & masks.keys() & labels.keys()
    for key in sorted(all_keys - complete_keys):
        missing = [
            kind
            for kind, index in (("RGB", images), ("mask", masks), ("label", labels))
            if key not in index
        ]
        errors.append(f"SynFoot V2 sample {key!r} is missing {', '.join(missing)}")

    samples = [
        _SynFootV2Sample(key=key, image=images[key], mask=masks[key], label=labels[key])
        for key in sorted(complete_keys)
    ]
    if not samples:
        errors.append("No complete SynFoot V2 RGB/mask/label samples found")
    return samples, errors, warnings


def _load_group_overrides(path: str | os.PathLike[str] | None) -> dict[str, str]:
    if path is None:
        return {}
    overrides: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source_sample_id", "group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise FootDataError("Group override CSV needs source_sample_id and group_id columns")
        for row in reader:
            source_sample_id = (row.get("source_sample_id") or "").strip().replace("\\", "/")
            group_id = (row.get("group_id") or "").strip()
            if not source_sample_id or not group_id:
                raise FootDataError("Group override CSV contains an empty key or group_id")
            if source_sample_id in overrides:
                raise FootDataError(f"Duplicate group override for {source_sample_id!r}")
            overrides[source_sample_id] = group_id
    return overrides


def _load_synfoot_labels(root: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        foot = value.get("foot")
        foot_id = value.get("foot-id") or value.get("foot_id")
        if isinstance(foot, dict):
            foot_id = foot_id or foot.get("name") or foot.get("id")
        if not isinstance(foot_id, (str, int)):
            continue
        labels[_canonical_key(path, root)] = str(foot_id)
    return labels


def _synfoot_v2_vector3(
    value: object,
    field: str,
    path: Path,
) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise FootDataError(f"SynFoot V2 label {path} needs a three-number {field}")
    parsed: list[float] = []
    for component in value:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise FootDataError(f"SynFoot V2 label {path} has a non-numeric {field}")
        number = float(component)
        if not math.isfinite(number):
            raise FootDataError(f"SynFoot V2 label {path} has a non-finite {field}")
        parsed.append(number)
    return parsed[0], parsed[1], parsed[2]


def _load_synfoot_v2_label(path: Path) -> _SynFootV2Label:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FootDataError(f"Cannot read SynFoot V2 label {path}: {error}") from error
    if not isinstance(value, dict):
        raise FootDataError(f"SynFoot V2 label {path} must contain an object")

    foot_id = value.get("foot-id")
    if not isinstance(foot_id, str) or not foot_id.strip():
        raise FootDataError(f"SynFoot V2 label {path} needs a non-empty string foot-id")
    camera_position = _synfoot_v2_vector3(
        value.get("camera_position"), "camera_position", path
    )
    camera_look_at = _synfoot_v2_vector3(
        value.get("camera_look_at"), "camera_look_at", path
    )
    foot_poses = value.get("foot_poses")
    if foot_poses is not None and not isinstance(foot_poses, dict):
        raise FootDataError(f"SynFoot V2 label {path} foot_poses must be an object")
    foot_shape = value.get("foot_shape")
    if foot_shape is not None and not isinstance(foot_shape, dict):
        raise FootDataError(f"SynFoot V2 label {path} foot_shape must be an object")
    viewpoint = tuple(
        position - look_at
        for position, look_at in zip(camera_position, camera_look_at, strict=True)
    )
    viewpoint_length = math.sqrt(sum(component * component for component in viewpoint))
    if viewpoint_length <= 1e-12:
        raise FootDataError(
            f"SynFoot V2 label {path} camera_position and camera_look_at must differ"
        )
    if viewpoint[2] <= 0:
        raise FootDataError(
            f"SynFoot V2 label {path} camera_position must be above camera_look_at"
        )
    horizontal_length = math.hypot(viewpoint[0], viewpoint[1])
    camera_tilt_deg = math.degrees(math.atan2(horizontal_length, viewpoint[2]))
    camera_azimuth_deg = (
        0.0
        if horizontal_length <= 1e-12
        else math.degrees(math.atan2(viewpoint[1], viewpoint[0])) % 360.0
    )
    return _SynFootV2Label(
        foot_id=foot_id.strip(),
        camera_position=camera_position,
        camera_look_at=camera_look_at,
        camera_tilt_deg=camera_tilt_deg,
        camera_azimuth_deg=camera_azimuth_deg,
        foot_poses=dict(foot_poses) if foot_poses is not None else None,
        foot_shape=dict(foot_shape) if foot_shape is not None else None,
    )


def _load_footnet_naming_map(root: Path) -> dict[str, str]:
    """Index publisher filenames for provenance only.

    The numeric suffixes are not documented participant identifiers, so they
    must never make ``group_resolved`` true or be used for held-out splits.
    """

    original_names: dict[str, str] = {}
    ambiguous: set[str] = set()
    for mapping_path in sorted(root.rglob("naming_map.csv")):
        try:
            with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"old_image", "new_image"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    continue
                for row in reader:
                    old_image = (row.get("old_image") or "").strip()
                    new_image = (row.get("new_image") or "").strip().casefold()
                    if not new_image or not old_image:
                        continue
                    previous = original_names.get(new_image)
                    if previous is not None and previous != old_image:
                        ambiguous.add(new_image)
                    else:
                        original_names[new_image] = old_image
        except OSError:
            continue
    for key in ambiguous:
        original_names.pop(key, None)
    return original_names


def _subject_from_filename(stem: str, custom_regex: re.Pattern[str] | None) -> str | None:
    patterns = (custom_regex,) if custom_regex is not None else SUBJECT_PATTERNS
    for pattern in patterns:
        match = pattern.search(stem)
        if not match:
            continue
        try:
            value = match.group("subject")
        except (IndexError, KeyError):
            value = match.group(1) if match.lastindex else match.group(0)
        if value:
            return str(value)
    return None


def _resolve_group(
    spec: SourceSpec,
    pair: _Pair,
    source_root: Path,
    overrides: dict[str, str],
    synfoot_labels: dict[str, str],
    group_pattern: re.Pattern[str] | None,
    sample_digest: str,
) -> tuple[str, bool, str]:
    source_sample_id = pair.image.relative_to(source_root).as_posix()
    override = overrides.get(source_sample_id) or overrides.get(pair.key)
    if override:
        token = override
        method = "override"
    elif spec.synthetic:
        token = synfoot_labels.get(pair.key)
        method = "synfoot-label"
    else:
        token = _subject_from_filename(pair.image.stem, group_pattern)
        method = "filename-pattern"

    if token:
        # Store a stable pseudonymous key, not a raw participant/scan identifier.
        return f"{spec.key}:group-{_digest(str(token), 16)}", True, method
    return f"{spec.key}:unresolved-{sample_digest}", False, "unresolved"


def _infer_view(pair: _Pair, root: Path) -> str:
    tokens = re.split(
        r"[^a-z0-9]+",
        pair.image.relative_to(root).as_posix().casefold(),
    )
    for token in tokens:
        if token in VIEW_ALIASES:
            return VIEW_ALIASES[token]
        if token.endswith(("top", "topview")):
            return "dorsal"
        if token.endswith(("sole", "soleview")):
            return "plantar"
        if token.endswith(("side", "sideview")):
            # FootNet's paper defines its side capture as the medial view.
            # An explicit generic "lateral" token is still handled by
            # VIEW_ALIASES above.
            return "medial"
    return "unknown"


def _infer_laterality(pair: _Pair, root: Path) -> str:
    tokens = set(
        token
        for token in re.split(
            r"[^a-z0-9]+", pair.image.relative_to(root).as_posix().casefold()
        )
        if token
    )
    if tokens & {"left", "lf", "lfoot"}:
        return "left"
    if tokens & {"right", "rt", "rfoot"}:
        return "right"
    if any(token.startswith("left") for token in tokens):
        return "left"
    if any(token.startswith("right") for token in tokens):
        return "right"
    return "unknown"


def _atomic_save_png(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".png", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", compress_level=6, optimize=False)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_decoded_size(image: Image.Image, path: Path, kind: str) -> None:
    """Reject implausibly large decoded images before ``Image.load()``."""

    width, height = image.size
    if width <= 0 or height <= 0:
        raise FootDataError(
            f"{kind.capitalize()} has invalid decoded dimensions at {path}: {width}x{height}"
        )
    if width > MAX_DECODED_DIMENSION or height > MAX_DECODED_DIMENSION:
        raise FootDataError(
            f"{kind.capitalize()} decoded dimension limit exceeded at {path}: "
            f"{width}x{height}; maximum side is {MAX_DECODED_DIMENSION} pixels"
        )
    pixels = width * height
    if pixels > MAX_DECODED_PIXELS:
        raise FootDataError(
            f"{kind.capitalize()} decoded pixel limit exceeded at {path}: "
            f"{pixels} pixels; maximum is {MAX_DECODED_PIXELS}"
        )


def _make_record(
    spec: SourceSpec,
    pair: _Pair,
    source_root: Path,
    output_root: Path,
    overrides: dict[str, str],
    synfoot_labels: dict[str, str],
    footnet_names: dict[str, str],
    group_pattern: re.Pattern[str] | None,
) -> dict[str, Any]:
    sample_digest = _digest(f"{spec.source_id}\0{pair.key}")
    sample_id = f"{spec.key}:{sample_digest}"
    source_sample_id = pair.image.relative_to(source_root).as_posix()
    image_output = output_root / "images" / f"{sample_digest}.png"
    mask_output = output_root / "masks" / f"{sample_digest}.png"
    original_image_sha256 = sha256_file(pair.image)
    original_mask_sha256 = sha256_file(pair.mask)

    try:
        with Image.open(pair.image) as image_source:
            _validate_decoded_size(image_source, pair.image, "image")
            image_source.load()
            # Phone photos can store portrait orientation only in EXIF while
            # FootNet masks are saved in display orientation. Apply it before
            # removing metadata so dimensions and pixels align with the mask.
            image = ImageOps.exif_transpose(image_source).convert("RGB")
        with Image.open(pair.mask) as mask_source:
            _validate_decoded_size(mask_source, pair.mask, "mask")
            mask_source.load()
            mask_grayscale = np.asarray(mask_source.convert("L"))
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise FootDataError(f"Cannot decode pair {pair.key!r}: {error}") from error

    transformations: list[str] = []
    mask_size = (int(mask_grayscale.shape[1]), int(mask_grayscale.shape[0]))
    # FootNet v1.0 righttop_004 has landscape pixels, no EXIF orientation, and
    # a portrait publisher mask. Apply the verified clockwise correction only
    # when both its filename and release SHA-256 match. Reversed dimensions on
    # any other sample remain a QA failure rather than triggering a guess.
    expected_rotation_name = FOOTNET_CLOCKWISE_ROTATIONS.get(original_image_sha256)
    if (
        spec.key == "footnet"
        and expected_rotation_name == pair.image.name.casefold()
        and image.size != mask_size
        and image.size == mask_size[::-1]
    ):
        image = image.transpose(Image.Transpose.ROTATE_270)
        transformations.append("footnet_rotate_90_clockwise_for_mask_alignment")

    normalized_mask, already_binary = normalize_mask(mask_grayscale)
    qa = analyze_mask(normalized_mask, already_binary=already_binary).to_dict()
    if transformations:
        qa["flags"].append("image_orientation_corrected")
    if image.size != mask_size:
        qa["valid"] = False
        qa["flags"].append("image_mask_dimension_mismatch")

    group_id, group_resolved, group_method = _resolve_group(
        spec,
        pair,
        source_root,
        overrides,
        synfoot_labels,
        group_pattern,
        sample_digest,
    )
    if not group_resolved:
        qa["flags"].append("missing_group_id_train_only")
    if spec.license_id.startswith("LicenseRef-"):
        qa["flags"].append("verify_archive_license_terms")

    _atomic_save_png(image, image_output)
    _atomic_save_png(Image.fromarray(normalized_mask, mode="L"), mask_output)
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "task": spec.task,
        "source_id": spec.source_id,
        "source_sample_id": source_sample_id,
        "image_path": image_output.relative_to(output_root).as_posix(),
        "mask_path": mask_output.relative_to(output_root).as_posix(),
        "image_sha256": sha256_file(image_output),
        "mask_sha256": sha256_file(mask_output),
        "width": image.width,
        "height": image.height,
        "group_id": group_id,
        "group_resolved": group_resolved,
        "group_method": group_method,
        "capture_id": f"{spec.key}:capture-{_digest(pair.key, 16)}",
        "view": _infer_view(pair, source_root),
        "laterality": _infer_laterality(pair, source_root),
        "synthetic": spec.synthetic,
        "license_id": spec.license_id,
        "split": None,
        "qa": qa,
        "provenance": {
            "original_image_path": source_sample_id,
            "original_mask_path": pair.mask.relative_to(source_root).as_posix(),
            "publisher_original_image_name": footnet_names.get(pair.image.name.casefold()),
            "original_image_sha256": original_image_sha256,
            "original_mask_sha256": original_mask_sha256,
            "importer_version": __version__,
            "normalization_transforms": transformations,
        },
    }


def _make_in_place_synfoot_v2_record(
    spec: SourceSpec,
    sample: _SynFootV2Sample,
    source_root: Path,
) -> dict[str, Any]:
    sample_digest = _digest(f"{spec.source_id}\0{sample.key}")
    sample_id = f"{spec.key}:{sample_digest}"
    source_sample_id = sample.image.relative_to(source_root).as_posix()
    label = _load_synfoot_v2_label(sample.label)
    image_sha256 = sha256_file(sample.image)
    mask_sha256 = sha256_file(sample.mask)
    label_sha256 = sha256_file(sample.label)

    try:
        with Image.open(sample.image) as image_source:
            _validate_decoded_size(image_source, sample.image, "image")
            image_source.load()
            image_size = image_source.size
            image_mode = image_source.mode
        with Image.open(sample.mask) as mask_source:
            _validate_decoded_size(mask_source, sample.mask, "mask")
            mask_source.load()
            mask_grayscale = np.asarray(mask_source.convert("L"))
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise FootDataError(
            f"Cannot decode SynFoot V2 sample {sample.key!r}: {error}"
        ) from error

    already_binary = bool(
        not np.any((mask_grayscale != 0) & (mask_grayscale != 255))
    )
    normalized_mask = (mask_grayscale >= 128).astype(np.uint8) * 255
    qa = analyze_mask(normalized_mask, already_binary=already_binary).to_dict()
    mask_size = (int(mask_grayscale.shape[1]), int(mask_grayscale.shape[0]))
    if image_size != mask_size:
        qa["valid"] = False
        qa["flags"].append("image_mask_dimension_mismatch")
    qa["flags"].append("in_place_assets_not_normalized")
    qa["flags"].append("verify_archive_license_terms")
    geometry_eligible = bool(
        qa["valid"]
        and not qa["touches_border"]
        and "fragmented_mask" not in qa["flags"]
    )
    qa["geometry_eligible"] = geometry_eligible

    subsets: list[str] = []
    if qa["valid"]:
        subsets.append("segmentation-valid")
    if geometry_eligible and label.camera_tilt_deg <= 35.0:
        subsets.append("measurement-top35")
    if geometry_eligible and label.camera_tilt_deg <= 25.0:
        subsets.append("measurement-top25")

    group_id = f"{spec.key}:group-{_digest(label.foot_id, 16)}"
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "task": spec.task,
        "source_id": spec.source_id,
        "source_sample_id": source_sample_id,
        "image_path": source_sample_id,
        "mask_path": sample.mask.relative_to(source_root).as_posix(),
        "label_path": sample.label.relative_to(source_root).as_posix(),
        "image_sha256": image_sha256,
        "mask_sha256": mask_sha256,
        "label_sha256": label_sha256,
        "width": image_size[0],
        "height": image_size[1],
        "group_id": group_id,
        "group_resolved": True,
        "group_method": "synfoot-v2-foot-id",
        "capture_id": f"{spec.key}:capture-{_digest(sample.key, 16)}",
        "view": "unknown",
        "laterality": "unknown",
        "synthetic": True,
        "license_id": spec.license_id,
        "split": None,
        "subsets": subsets,
        "camera": {
            "position": list(label.camera_position),
            "look_at": list(label.camera_look_at),
            "tilt_deg": label.camera_tilt_deg,
            "tilt_from_vertical_deg": label.camera_tilt_deg,
            "azimuth_deg": label.camera_azimuth_deg,
        },
        "preprocessing": {
            "image_decode_channels": 3,
            "source_image_mode": image_mode,
            "mask_foreground_threshold_gte": 128,
        },
        "foot_poses": label.foot_poses,
        "foot_shape_coefficients": label.foot_shape,
        "qa": qa,
        "provenance": {
            "original_image_path": source_sample_id,
            "original_mask_path": sample.mask.relative_to(source_root).as_posix(),
            "original_label_path": sample.label.relative_to(source_root).as_posix(),
            "original_image_sha256": image_sha256,
            "original_mask_sha256": mask_sha256,
            "original_label_sha256": label_sha256,
            "importer_version": __version__,
            "normalization_transforms": [],
        },
    }


def _mark_exact_duplicates(records: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_hash.setdefault(record["image_sha256"], []).append(record)
    for image_hash, duplicates in by_hash.items():
        if len(duplicates) < 2:
            continue
        cluster_id = f"sha256:{image_hash}"
        for record in duplicates:
            record["duplicate_cluster_id"] = cluster_id
            if "exact_duplicate_image" not in record["qa"]["flags"]:
                record["qa"]["flags"].append("exact_duplicate_image")


def import_dataset(
    source_key: str,
    input_directory: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    lock_path: str | os.PathLike[str] | None = None,
    group_overrides: str | os.PathLike[str] | None = None,
    group_regex: str | None = None,
    license_accepted: bool = False,
    acquisition: dict[str, Any] | None = None,
    in_place: bool = False,
) -> ImportResult:
    """Normalize or in-place index a dataset and write its manifest plus provenance lock."""

    if not license_accepted:
        raise FootDataError(
            "Import requires explicit license acceptance (--accept-license); "
            "review the source landing page and archive terms first"
        )
    spec = get_source(source_key)
    source_root = Path(input_directory).resolve()
    destination = Path(output_root).resolve()
    if in_place and spec.key != "synfoot-v2":
        raise FootDataError("--in-place is supported only for synfoot-v2")
    if in_place and source_root != destination:
        raise FootDataError(
            "SynFoot V2 --in-place requires --output-root to equal --input-dir"
        )
    if in_place and (group_overrides is not None or group_regex is not None):
        raise FootDataError(
            "SynFoot V2 --in-place reads groups from foot-id and does not accept group overrides"
        )
    output_manifest = (
        Path(manifest_path).resolve() if manifest_path else destination / "samples.jsonl"
    )
    output_lock = Path(lock_path).resolve() if lock_path else destination / "dataset.lock.json"
    if output_manifest == output_lock:
        raise FootDataError("Manifest and dataset lock must use different paths")
    if output_manifest.parent != destination:
        raise FootDataError(
            "Import manifest must be directly under --output-root so relative asset paths "
            f"have an unambiguous base: {output_manifest}"
        )
    if output_lock.parent != destination:
        raise FootDataError(
            "Import dataset lock must be directly under --output-root: "
            f"{output_lock}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    overrides = _load_group_overrides(group_overrides)
    try:
        compiled_group_regex = re.compile(group_regex) if group_regex else None
    except re.error as error:
        raise FootDataError(f"Invalid --group-regex: {error}") from error
    records: list[dict[str, Any]] = []
    if in_place:
        samples, errors, warnings = _discover_synfoot_v2_samples(source_root)
        for sample in samples:
            try:
                records.append(
                    _make_in_place_synfoot_v2_record(spec, sample, source_root)
                )
            except FootDataError as error:
                errors.append(str(error))
    else:
        synfoot_labels = _load_synfoot_labels(source_root) if spec.synthetic else {}
        footnet_names = _load_footnet_naming_map(source_root) if spec.key == "footnet" else {}
        pairs, errors, warnings = discover_pairs(source_root)
        for pair in pairs:
            try:
                records.append(
                    _make_record(
                        spec,
                        pair,
                        source_root,
                        destination,
                        overrides,
                        synfoot_labels,
                        footnet_names,
                        compiled_group_regex,
                    )
                )
            except FootDataError as error:
                errors.append(str(error))
    _mark_exact_duplicates(records)
    write_manifest(records, output_manifest)

    lock = {
        "schema_version": 1,
        "source": spec.to_dict(),
        "license_receipt": {
            "accepted": True,
            "accepted_at": datetime.now(UTC).isoformat(),
            "license_id": spec.license_id,
            "license_url": spec.license_url,
        },
        "acquisition": acquisition
        or {"kind": "local-directory-in-place" if in_place else "local-directory"},
        "importer_version": __version__,
        "record_count": len(records),
        "valid_record_count": sum(record["qa"]["valid"] for record in records),
        "unresolved_group_count": sum(not record["group_resolved"] for record in records),
        "manifest_sha256": sha256_file(output_manifest),
        "errors": errors,
        "warnings": warnings,
    }
    if in_place:
        lock["metadata_notes"] = {
            "foot_shape_coefficients": (
                "Publisher-provided shape coefficients; they are not physical dimensions "
                "and must not be interpreted as millimetres."
            )
        }
    write_json(lock, output_lock)
    return ImportResult(
        records=records,
        manifest_path=output_manifest,
        lock_path=output_lock,
        errors=errors,
        warnings=warnings,
    )


def import_archive(
    source_key: str,
    archive_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    archive_limits: ArchiveLimits | None = None,
    **options: Any,
) -> ImportResult:
    """Verify, safely extract, then import a local publisher archive."""

    if options.get("in_place"):
        raise FootDataError("--in-place requires --input-dir and cannot import an archive")

    archive = Path(archive_path).resolve()
    actual_sha256 = sha256_file(archive)
    if expected_sha256 and actual_sha256.casefold() != expected_sha256.casefold():
        raise FootDataError(
            f"Archive SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    with tempfile.TemporaryDirectory(prefix="footdata-import-") as temporary:
        extracted = safe_extract_zip(archive, Path(temporary) / "extracted", archive_limits)
        acquisition = {
            "kind": "local-archive",
            "filename": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": actual_sha256,
        }
        return import_dataset(
            source_key,
            extracted,
            output_root,
            acquisition=acquisition,
            **options,
        )
