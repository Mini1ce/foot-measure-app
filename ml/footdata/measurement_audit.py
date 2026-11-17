"""Visual review and explicit filtering for measurement-oriented masks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps

from ml.footseg.geometry import mask_dimensions

from .errors import FootDataError
from .manifest import sha256_file, write_json, write_manifest
from .qa import analyze_mask, normalize_mask


REVIEW_COLUMNS = (
    "sample_id",
    "decision",
    "notes",
    "suggested_decision",
    "reasons",
    "split",
    "view",
    "image_path",
    "mask_path",
)
FINAL_DECISIONS = {"include", "exclude"}


@dataclass(frozen=True)
class MeasurementAuditPolicy:
    """Configurable triage limits; these are review aids, not accuracy claims."""

    minimum_foreground_fraction: float = 0.015
    maximum_foreground_fraction: float = 0.75
    minimum_largest_component_fraction: float = 0.98
    minimum_length_width_ratio: float = 1.35
    maximum_length_width_ratio: float = 4.5
    thumbnail_size: int = 320

    def validate(self) -> None:
        fractions = (
            self.minimum_foreground_fraction,
            self.maximum_foreground_fraction,
            self.minimum_largest_component_fraction,
        )
        if any(not np.isfinite(value) for value in fractions):
            raise FootDataError("Measurement audit fractions must be finite")
        if not 0 <= self.minimum_foreground_fraction < self.maximum_foreground_fraction <= 1:
            raise FootDataError("Foreground fraction limits must satisfy 0 <= min < max <= 1")
        if not 0 < self.minimum_largest_component_fraction <= 1:
            raise FootDataError("Largest-component fraction must be in (0, 1]")
        if (
            not np.isfinite(self.minimum_length_width_ratio)
            or not np.isfinite(self.maximum_length_width_ratio)
            or self.minimum_length_width_ratio <= 0
            or self.minimum_length_width_ratio >= self.maximum_length_width_ratio
        ):
            raise FootDataError("Length/width ratio limits must be finite and increasing")
        if self.thumbnail_size < 96 or self.thumbnail_size > 1024:
            raise FootDataError("Thumbnail size must be between 96 and 1024 pixels")


def _safe_relative_path(raw_path: object, *, field: str) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path:
        raise FootDataError(f"{field} must be a non-empty relative path")
    normalized = raw_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in path.parts):
        raise FootDataError(f"Unsafe {field}: {raw_path!r}")
    return path


def _resolve_asset(data_root: Path, raw_path: object, *, field: str) -> Path:
    relative = _safe_relative_path(raw_path, field=field)
    root = data_root.resolve()
    resolved = data_root.joinpath(*relative.parts).resolve()
    if not resolved.is_relative_to(root):
        raise FootDataError(f"{field} escapes data root: {relative.as_posix()}")
    if not resolved.is_file():
        raise FootDataError(f"Missing {field}: {relative.as_posix()}")
    return resolved


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(binary)
    height, width = binary.shape
    for offset_y in range(3):
        for offset_x in range(3):
            eroded &= padded[offset_y : offset_y + height, offset_x : offset_x + width]
    return binary & ~eroded


def _fit_panel(image: Image.Image, size: int, *, background: str) -> Image.Image:
    contained = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), background)
    panel.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return panel


def _write_thumbnail(
    image: Image.Image,
    binary_mask: np.ndarray,
    destination: Path,
    *,
    size: int,
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != binary_mask.shape:
        raise FootDataError(
            f"Image/mask dimensions differ: image={rgb.shape[:2]}, mask={binary_mask.shape}"
        )
    overlay = rgb.astype(np.float32)
    foreground = binary_mask.astype(bool)
    overlay[foreground] = overlay[foreground] * 0.68 + np.array([34, 197, 94]) * 0.32
    overlay[_inner_boundary(foreground)] = np.array([239, 68, 68])
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    mask_image = Image.fromarray(foreground.astype(np.uint8) * 255, mode="L").convert("RGB")
    gap = 12
    canvas = Image.new("RGB", (size * 2 + gap, size), "#111827")
    canvas.paste(_fit_panel(overlay_image, size, background="#111827"), (0, 0))
    canvas.paste(_fit_panel(mask_image, size, background="#000000"), (size + gap, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=88, optimize=True)


def _reason_status(reasons: list[tuple[str, str]]) -> str:
    severities = {severity for severity, _ in reasons}
    if "exclude" in severities:
        return "exclude"
    if "review" in severities:
        return "review"
    return "include"


def analyze_measurement_record(
    record: dict[str, Any],
    *,
    data_root: Path,
    policy: MeasurementAuditPolicy,
) -> tuple[dict[str, object], Image.Image, np.ndarray]:
    """Return deterministic triage metadata plus source image and binary mask."""

    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise FootDataError("Measurement audit record lacks sample_id")
    image_path = _resolve_asset(data_root, record.get("image_path"), field="image_path")
    mask_path = _resolve_asset(data_root, record.get("mask_path"), field="mask_path")
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    with Image.open(mask_path) as source_mask:
        raw_mask = np.asarray(source_mask.convert("L"))
    normalized, already_binary = normalize_mask(raw_mask)
    binary = normalized > 0
    qa = analyze_mask(normalized, already_binary=already_binary)
    reasons: list[tuple[str, str]] = []

    if not qa.valid:
        reasons.append(("exclude", "invalid-mask"))
    if qa.component_count > 1 and qa.largest_component_fraction < policy.minimum_largest_component_fraction:
        reasons.append(("exclude", "fragmented-mask"))
    elif qa.component_count > 1:
        reasons.append(("review", "minor-detached-components"))
    if qa.touches_border:
        reasons.append(("review", "mask-touches-border"))
    if qa.foreground_fraction < policy.minimum_foreground_fraction:
        reasons.append(("review", "foreground-too-small"))
    if qa.foreground_fraction > policy.maximum_foreground_fraction:
        reasons.append(("review", "foreground-too-large"))
    if image.size != (raw_mask.shape[1], raw_mask.shape[0]):
        reasons.append(("exclude", "image-mask-size-mismatch"))

    dimensions = mask_dimensions(binary)
    ratio: float | None = None
    if dimensions is None:
        reasons.append(("review", "measurement-geometry-unavailable"))
    elif dimensions.width_px > 0:
        ratio = dimensions.length_px / dimensions.width_px
        if ratio < policy.minimum_length_width_ratio:
            reasons.append(("review", "silhouette-too-round"))
        elif ratio > policy.maximum_length_width_ratio:
            reasons.append(("review", "silhouette-too-elongated"))

    status = _reason_status(reasons)
    reason_values = sorted({reason for _, reason in reasons})
    item: dict[str, object] = {
        "sample_id": sample_id,
        "split": record.get("split"),
        "view": record.get("view"),
        "suggested_decision": status,
        "reasons": reason_values,
        "image_path": str(record.get("image_path", "")),
        "mask_path": str(record.get("mask_path", "")),
        "width": image.width,
        "height": image.height,
        "foreground_fraction": qa.foreground_fraction,
        "component_count": qa.component_count,
        "largest_component_fraction": qa.largest_component_fraction,
        "touches_border": qa.touches_border,
        "bbox_xywh": qa.bbox_xywh,
        "length_px": dimensions.length_px if dimensions else None,
        "width_px": dimensions.width_px if dimensions else None,
        "length_width_ratio": ratio,
        "geometry_confidence": dimensions.confidence if dimensions else None,
    }
    return item, image, binary


def _write_review_csv(items: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    **item,
                    "decision": "",
                    "notes": "",
                    "reasons": "|".join(str(value) for value in item["reasons"]),
                }
            )
    temporary.replace(destination)


def _html_document(items: list[dict[str, object]], policy: MeasurementAuditPolicy) -> str:
    safe_json = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    safe_json = safe_json.replace("<", "\\u003c").replace("&", "\\u0026")
    policy_json = escape(json.dumps(asdict(policy), sort_keys=True), quote=True)
    return f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Foot mask measurement audit</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e5e7eb}}
header{{position:sticky;top:0;z-index:2;padding:16px;background:#111827;border-bottom:1px solid #334155}}
.toolbar{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}button,select,input{{font:inherit}}
button{{padding:8px 12px;border:0;border-radius:8px;background:#2563eb;color:white;cursor:pointer}}
.muted{{color:#94a3b8;font-size:13px}}main{{padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}}
.card{{background:#1e293b;border:1px solid #334155;border-radius:12px;overflow:hidden}}.card img{{width:100%;display:block;background:#111827}}
.body{{padding:12px}}.id{{font-family:ui-monospace,monospace;word-break:break-all;font-size:12px}}
.meta{{font-size:13px;color:#cbd5e1;margin:8px 0}}.reasons{{color:#fbbf24;font-size:13px;min-height:20px}}
.controls{{display:grid;grid-template-columns:130px 1fr;gap:8px;margin-top:10px}}.controls input,.controls select{{min-width:0;padding:7px;border-radius:6px;border:1px solid #475569;background:#0f172a;color:#e5e7eb}}
.include{{border-color:#22c55e}}.exclude{{border-color:#ef4444}}.review{{border-color:#f59e0b}}
</style>
</head>
<body>
<header>
<div class="toolbar"><strong>Foot mask measurement audit</strong><button id="download">ดาวน์โหลด measurement-review.csv</button><select id="filter"><option value="all">ทั้งหมด</option><option value="include">แนะนำ include</option><option value="review">ต้องตรวจ</option><option value="exclude">แนะนำ exclude</option></select><span id="count"></span></div>
<div class="muted">ซ้าย: ภาพซ้อน mask (ขอบแดง) ขวา: mask ดิบ • เกณฑ์ช่วยคัดเท่านั้น ไม่ใช่ผลรับรองความแม่นยำ • policy={policy_json}</div>
</header>
<main id="grid"></main>
<script id="audit-data" type="application/json">{safe_json}</script>
<script>
const rows=JSON.parse(document.getElementById('audit-data').textContent);const grid=document.getElementById('grid');
function html(value){{return String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');}}
function render(){{const filter=document.getElementById('filter').value;grid.textContent='';let visible=0;for(const row of rows){{if(filter!=='all'&&row.suggested_decision!==filter)continue;visible++;const card=document.createElement('article');card.className='card '+row.suggested_decision;card.innerHTML=`<img loading="lazy" src="${{row.thumbnail}}" alt="mask audit"><div class="body"><div class="id">${{html(row.sample_id)}}</div><div class="meta">split=${{html(row.split??'-')}} view=${{html(row.view??'-')}} • foreground=${{(row.foreground_fraction*100).toFixed(2)}}% • components=${{row.component_count}} • L/W=${{row.length_width_ratio==null?'-':row.length_width_ratio.toFixed(2)}}</div><div class="reasons">${{html(row.reasons.length?row.reasons.join(', '):'ผ่าน auto checks')}}</div><div class="controls"><select><option value="">ยังไม่ตัดสิน</option><option value="include">include</option><option value="exclude">exclude</option></select><input placeholder="หมายเหตุ"></div></div>`;const decision=card.querySelector('select');const notes=card.querySelector('input');decision.value=row.decision??'';notes.value=row.notes??'';decision.addEventListener('change',()=>{{row.decision=decision.value;}});notes.addEventListener('input',()=>{{row.notes=notes.value;}});grid.appendChild(card);}}document.getElementById('count').textContent=visible+'/'+rows.length;}}
function csvCell(value){{const text=String(value??'');return '"'+text.replaceAll('"','""')+'"';}}
document.getElementById('download').addEventListener('click',()=>{{const header={json.dumps(list(REVIEW_COLUMNS), ensure_ascii=False)};const lines=[header.map(csvCell).join(',')];for(const row of rows){{const values=[row.sample_id,row.decision??'',row.notes??'',row.suggested_decision,row.reasons.join('|'),row.split??'',row.view??'',row.image_path,row.mask_path];lines.push(values.map(csvCell).join(','));}}const blob=new Blob(['\\ufeff'+lines.join('\\r\\n')],{{type:'text/csv;charset=utf-8'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='measurement-review.csv';link.click();URL.revokeObjectURL(link.href);}});
document.getElementById('filter').addEventListener('change',render);render();
</script>
</body>
</html>
"""


def build_measurement_audit(
    records: Iterable[dict[str, Any]],
    *,
    data_root: Path,
    output_directory: Path,
    views: set[str] | None = None,
    splits: set[str] | None = None,
    policy: MeasurementAuditPolicy = MeasurementAuditPolicy(),
) -> dict[str, object]:
    """Generate thumbnails, HTML, CSV template, and machine-readable summary."""

    policy.validate()
    selected = [
        record
        for record in records
        if (views is None or record.get("view") in views)
        and (splits is None or record.get("split") in splits)
    ]
    if not selected:
        raise FootDataError("No records matched measurement audit filters")
    output_directory.mkdir(parents=True, exist_ok=True)
    thumbnail_directory = output_directory / "thumbnails"
    items: list[dict[str, object]] = []
    for record in sorted(selected, key=lambda value: str(value.get("sample_id", ""))):
        item, image, binary = analyze_measurement_record(
            record, data_root=data_root, policy=policy
        )
        thumbnail_name = hashlib.sha256(str(item["sample_id"]).encode("utf-8")).hexdigest()[:24] + ".jpg"
        thumbnail_path = thumbnail_directory / thumbnail_name
        _write_thumbnail(
            image,
            binary,
            thumbnail_path,
            size=policy.thumbnail_size,
        )
        item["thumbnail"] = f"thumbnails/{thumbnail_name}"
        items.append(item)

    counts = Counter(str(item["suggested_decision"]) for item in items)
    reason_counts = Counter(
        str(reason) for item in items for reason in item.get("reasons", [])
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "record_count": len(items),
        "suggested_decisions": dict(sorted(counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "filters": {
            "views": sorted(views) if views else None,
            "splits": sorted(splits) if splits else None,
        },
        "policy": asdict(policy),
        "warning": (
            "Automatic decisions are triage suggestions only. Explicit human include/exclude "
            "decisions are required before creating a reviewed manifest."
        ),
    }
    review_csv = output_directory / "measurement-review-template.csv"
    report_html = output_directory / "index.html"
    items_json = output_directory / "items.json"
    _write_review_csv(items, review_csv)
    report_html.write_text(_html_document(items, policy), encoding="utf-8", newline="\n")
    write_json(items, items_json)
    write_json(summary, output_directory / "summary.json")
    return {
        **summary,
        "output_directory": str(output_directory),
        "report": str(report_html),
        "review_csv": str(review_csv),
        "items": str(items_json),
    }


def _read_review_rows(review_csv: Path) -> list[dict[str, str]]:
    with review_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames or "decision" not in reader.fieldnames:
            raise FootDataError("Review CSV requires sample_id and decision columns")
        rows = [dict(row) for row in reader]
    if not rows:
        raise FootDataError("Review CSV contains no decisions")
    return rows


def filter_reviewed_records(
    records: Iterable[dict[str, Any]],
    *,
    review_csv: Path,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    """Return records explicitly marked include and a reproducibility lock."""

    materialized = list(records)
    records_by_id = {str(record.get("sample_id")): record for record in materialized}
    rows = _read_review_rows(review_csv)
    decisions: dict[str, str] = {}
    unresolved: list[str] = []
    unknown: list[str] = []
    for line_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "").strip()
        decision = row.get("decision", "").strip().casefold()
        if not sample_id:
            raise FootDataError(f"Review CSV line {line_number} has empty sample_id")
        if sample_id in decisions:
            raise FootDataError(f"Duplicate review decision for {sample_id}")
        if sample_id not in records_by_id:
            unknown.append(sample_id)
        if decision not in FINAL_DECISIONS:
            unresolved.append(sample_id)
        decisions[sample_id] = decision
    if unknown:
        raise FootDataError(f"Review CSV contains unknown sample IDs: {sorted(unknown)[:10]}")
    if unresolved:
        raise FootDataError(
            "Every reviewed sample requires decision include or exclude; unresolved: "
            f"{sorted(unresolved)[:10]}"
        )
    included = [records_by_id[sample_id] for sample_id, decision in decisions.items() if decision == "include"]
    if not included:
        raise FootDataError("Review CSV includes zero samples")
    split_counts = Counter(str(record.get("split", "unset")) for record in included)
    if split_counts.get("train", 0) == 0 or split_counts.get("val", 0) == 0:
        raise FootDataError("Reviewed manifest requires at least one train and one val sample")
    ordered_ids = sorted(str(record["sample_id"]) for record in included)
    lock: dict[str, object] = {
        "schema_version": 1,
        "policy": "explicit-human-decisions-only",
        "review_csv_sha256": sha256_file(review_csv),
        "reviewed_rows": len(rows),
        "included_records": len(included),
        "excluded_records": len(rows) - len(included),
        "split_counts": dict(sorted(split_counts.items())),
        "included_sample_ids_sha256": hashlib.sha256(
            ("\n".join(ordered_ids) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    return included, lock


def write_reviewed_manifest(
    records: Iterable[dict[str, Any]],
    *,
    input_manifest: Path,
    review_csv: Path,
    output_manifest: Path,
    lock_path: Path,
) -> dict[str, object]:
    included, lock = filter_reviewed_records(records, review_csv=review_csv)
    lock = {
        **lock,
        "input_manifest": input_manifest.name,
        "input_manifest_sha256": sha256_file(input_manifest),
        "review_csv": review_csv.name,
        "output_manifest": output_manifest.name,
    }
    write_manifest(included, output_manifest)
    lock["output_manifest_sha256"] = sha256_file(output_manifest)
    write_json(lock, lock_path)
    return lock
