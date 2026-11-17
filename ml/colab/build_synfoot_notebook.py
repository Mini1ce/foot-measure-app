"""Build the SynFoot V2 pretrain -> FootNet fine-tune Colab notebook."""

from __future__ import annotations

import hashlib
import inspect
import json
import stat
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .build_notebook import code, embedded_source_archive, markdown


OUTPUT = Path(__file__).with_name("SynFoot_V2_Foot_Segmentation_Training.ipynb")


def canonical_zip_member_name(raw_name: str) -> str:
    """Return a canonical relative POSIX member name or reject it.

    ``PurePosixPath`` intentionally normalizes repeated separators and dot
    segments.  Comparing its result with the raw ZIP header is what prevents a
    superficially safe name such as ``rgb//x.png`` or ``rgb/./x.png`` from
    being accepted under a different spelling.
    """

    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise RuntimeError(f"unsafe ZIP member: {raw_name!r}")
    if "\\" in raw_name:
        raise RuntimeError(f"non-canonical ZIP member: {raw_name!r}")
    member = PurePosixPath(raw_name)
    if member.is_absolute() or any(
        part in {"", ".", ".."} or ":" in part for part in member.parts
    ):
        raise RuntimeError(f"unsafe ZIP member: {raw_name!r}")
    canonical = member.as_posix()
    if canonical != raw_name:
        raise RuntimeError(f"non-canonical ZIP member: {raw_name!r}")
    return canonical


def extract_verified_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination_root: Path,
    expected_sha256: str | None,
) -> str:
    """Extract one regular file with containment and optional hash checks."""

    name = canonical_zip_member_name(info.filename)
    unix_mode = info.external_attr >> 16
    if info.is_dir() or stat.S_ISDIR(unix_mode) or stat.S_ISLNK(unix_mode):
        raise RuntimeError(f"ZIP directories/symlinks are not allowed: {name!r}")

    root = destination_root.resolve()
    destination = destination_root.joinpath(*PurePosixPath(name).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    if not resolved_destination.is_relative_to(root):
        raise RuntimeError(f"ZIP member escapes extraction root: {name!r}")

    digest = hashlib.sha256()
    try:
        with archive.open(info, "r") as source, destination.open("xb") as target:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256.casefold():
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 extracted asset ไม่ตรง: {name}")
    return name


def validate_packaged_records(
    records: list[dict[str, Any]], package_lock: dict[str, Any]
) -> dict[str, Any]:
    """Validate package counts/groups and return the exact asset hash map."""

    if not isinstance(package_lock, dict):
        raise RuntimeError("package lock ต้องเป็น JSON object")
    lock_counts = package_lock.get("counts")
    lock_groups = package_lock.get("groups")
    if not isinstance(lock_counts, dict) or not isinstance(lock_groups, dict):
        raise RuntimeError("package lock ขาด counts/groups")

    counts = {"train": 0, "val": 0}
    groups: dict[str, set[str]] = {"train": set(), "val": set()}
    expected_assets: dict[str, str] = {}
    sample_ids: set[str] = set()
    reserved_paths = {".package-sha256", "samples.pretrain.jsonl"}
    required_subset = package_lock.get("subset")
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise RuntimeError(f"manifest record {index} ต้องเป็น JSON object")
        sample_id = record.get("sample_id")
        split = record.get("pretrain_split")
        group_id = record.get("group_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise RuntimeError(f"manifest record {index} ขาด sample_id")
        if sample_id in sample_ids:
            raise RuntimeError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if split not in counts:
            raise RuntimeError(f"{sample_id}: pretrain_split ต้องเป็น train/val")
        if not isinstance(group_id, str) or not group_id:
            raise RuntimeError(f"{sample_id}: ขาด group_id")
        if record.get("synthetic") is not True:
            raise RuntimeError(f"{sample_id}: package pretraining ต้องเป็น synthetic")
        if record.get("group_resolved") is not True:
            raise RuntimeError(f"{sample_id}: package ต้องมี resolved group")
        if record.get("split") != "train":
            raise RuntimeError(f"{sample_id}: main split ของ synthetic ต้องเป็น train")
        qa = record.get("qa")
        if not isinstance(qa, dict) or qa.get("valid") is not True:
            raise RuntimeError(f"{sample_id}: package ต้องมี qa.valid=true")
        subsets = record.get("subsets")
        if not isinstance(subsets, list) or any(
            not isinstance(subset, str) for subset in subsets
        ):
            raise RuntimeError(f"{sample_id}: subsets ไม่ถูกต้อง")
        if required_subset is not None and required_subset not in subsets:
            raise RuntimeError(f"{sample_id}: ไม่อยู่ใน subset {required_subset!r}")
        counts[split] += 1
        groups[split].add(group_id)

        for role in ("image", "mask", "label"):
            raw_path = record.get(f"{role}_path")
            raw_digest = record.get(f"{role}_sha256")
            if not isinstance(raw_path, str):
                raise RuntimeError(f"{sample_id}: ขาด {role}_path")
            path = canonical_zip_member_name(raw_path)
            if path in reserved_paths:
                raise RuntimeError(f"{sample_id}: asset path สงวนไว้: {path}")
            if (
                not isinstance(raw_digest, str)
                or len(raw_digest) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in raw_digest)
            ):
                raise RuntimeError(f"{sample_id}: {role}_sha256 ไม่ถูกต้อง")
            if path in expected_assets:
                raise RuntimeError(f"duplicate package asset path: {path}")
            expected_assets[path] = raw_digest.casefold()

    expected_counts: dict[str, int] = {}
    expected_groups: dict[str, set[str]] = {}
    for split in counts:
        value = lock_counts.get(split)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise RuntimeError(f"package lock count {split!r} ไม่ถูกต้อง")
        expected_counts[split] = value
        raw_groups = lock_groups.get(split)
        if not isinstance(raw_groups, list) or any(
            not isinstance(group, str) or not group for group in raw_groups
        ) or len(raw_groups) != len(set(raw_groups)):
            raise RuntimeError(f"package lock groups {split!r} ไม่ถูกต้อง")
        expected_groups[split] = set(raw_groups)

    if counts != expected_counts:
        raise RuntimeError(f"package counts ไม่ตรง: manifest={counts}, lock={expected_counts}")
    overlap = groups["train"] & groups["val"]
    if overlap:
        raise RuntimeError(f"train/val group leakage: {sorted(overlap)!r}")
    if groups != expected_groups:
        raise RuntimeError(
            "package groups ไม่ตรง: "
            f"manifest={{{', '.join(f'{key}: {sorted(value)!r}' for key, value in groups.items())}}}, "
            f"lock={{{', '.join(f'{key}: {sorted(value)!r}' for key, value in expected_groups.items())}}}"
        )
    return {
        "counts": counts,
        "groups": {key: sorted(value) for key, value in groups.items()},
        "expected_assets": expected_assets,
    }


SAFE_ZIP_HELPERS_SOURCE = "\n\n".join(
    inspect.getsource(function)
    for function in (canonical_zip_member_name, extract_verified_zip_member)
)
PACKAGE_VALIDATION_HELPER_SOURCE = inspect.getsource(validate_packaged_records)


def build_notebook() -> dict[str, object]:
    source_archive = embedded_source_archive()
    cells = [
        markdown(
            """# SynFoot V2 pretraining → FootNet fine-tuning\n
สมุดนี้ใช้ **SynFoot V2 เฉพาะ pretraining** แล้วปรับโมเดลด้วยภาพจริง FootNet ก่อน export สำหรับแอปวัดเท้า โดยแยก `foot-id` ระหว่าง synthetic train/validation เพื่อป้องกันภาพจากเท้าต้นแบบเดียวกันรั่วข้ามชุด\n
ก่อน Run all ให้อัปโหลดโฟลเดอร์ที่ได้จากคำสั่ง `package-pretrain` ไปไว้ที่ `MyDrive/foot-measure-training/synfoot-v2-package` แล้วเลือก Runtime → Change runtime type → **T4 GPU**\n
> SynFoot เป็นข้อมูลสังเคราะห์จากสแกนต้นแบบเพียงไม่กี่เท้า คะแนนจากชุดนี้ไม่ใช่หลักฐานความแม่นยำกับคนจริงหรือหน่วยมิลลิเมตร ผล release ยังต้องประเมินด้วยภาพมือถือจริง บัตรอ้างอิง และขนาดเท้าที่วัดจริง\n""",
            "intro",
        ),
        code(
            """# ตรวจว่า Colab เปิด GPU แล้ว
import os
import subprocess

gpu_check = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if gpu_check.returncode != 0:
    raise RuntimeError("ยังไม่ได้เปิด GPU: เลือก Runtime > Change runtime type > T4 GPU")
print("GPU พร้อม:", gpu_check.stdout.strip())
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
""",
            "gpu-check",
        ),
        code(
            """# ติดตั้งรุ่นที่รองรับ T4 และ pipeline นี้
import sys

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "tensorflow>=2.18,<2.21",
        "numpy>=1.26,<2.2",
        "pillow>=10,<13",
        "requests>=2.31,<3",
    ],
    check=True,
)
print("ติดตั้ง dependencies แล้ว")
""",
            "dependencies",
        ),
        code(
            f'''# แตก source code ที่ล็อกมากับ notebook แบบตรวจ canonical path ทีละไฟล์
import base64
import hashlib
import io
import stat
import zipfile
from pathlib import Path, PurePosixPath

{SAFE_ZIP_HELPERS_SOURCE}

SOURCE_ARCHIVE_B64 = "{source_archive}"
SOURCE_ARCHIVE_BYTES = base64.b64decode(SOURCE_ARCHIVE_B64, validate=True)
SOURCE_ARCHIVE_SHA256 = hashlib.sha256(SOURCE_ARCHIVE_BYTES).hexdigest()
PROJECT_DIR = Path(f"/content/foot-measure-app-{{SOURCE_ARCHIVE_SHA256[:16]}}")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(SOURCE_ARCHIVE_BYTES)) as archive:
    source_names: set[str] = set()
    for info in archive.infolist():
        canonical_name = canonical_zip_member_name(info.filename)
        unix_mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISDIR(unix_mode) or stat.S_ISLNK(unix_mode):
            raise RuntimeError(
                f"embedded source directories/symlinks are not allowed: {{canonical_name!r}}"
            )
        if canonical_name in source_names:
            raise RuntimeError(f"duplicate embedded source member: {{canonical_name}}")
        source_names.add(canonical_name)
        destination = PROJECT_DIR.joinpath(*PurePosixPath(canonical_name).parts)
        if destination.exists():
            if destination.is_file() and hashlib.sha256(
                destination.read_bytes()
            ).digest() == hashlib.sha256(archive.read(info)).digest():
                continue
            raise RuntimeError(f"embedded source destination conflict: {{canonical_name}}")
        extract_verified_zip_member(archive, info, PROJECT_DIR, None)
print("เตรียม source code แล้ว:", PROJECT_DIR)
print("embedded source SHA-256:", SOURCE_ARCHIVE_SHA256)
''',
            "embedded-code",
        ),
        code(
            """# เชื่อม Google Drive และกำหนดตำแหน่ง input/output
from google.colab import drive
import hashlib
import json
import shutil

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/foot-measure-training")
SYNFOOT_PACKAGE_DIR = DRIVE_ROOT / "synfoot-v2-package"
ARTIFACT_DIR = DRIVE_ROOT / "synfoot-v2-run"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print("SynFoot package:", SYNFOOT_PACKAGE_DIR)
print("ผลลัพธ์:", ARTIFACT_DIR)
""",
            "drive",
        ),
        code(
            """# ตรวจ SHA-256, manifest และ ZIP headers ก่อนแตก SynFoot ทีละไฟล์
import tempfile
from typing import Any

"""
            + PACKAGE_VALIDATION_HELPER_SOURCE
            + """

PACKAGE_LOCK_PATH = SYNFOOT_PACKAGE_DIR / "package.lock.json"
PACKAGE_MANIFEST_PATH = SYNFOOT_PACKAGE_DIR / "samples.pretrain.jsonl"
if not PACKAGE_LOCK_PATH.is_file() or not PACKAGE_MANIFEST_PATH.is_file():
    raise FileNotFoundError(
        "ไม่พบ SynFoot package ใน Drive; รัน package-pretrain ที่เครื่องแล้วอัปโหลดทั้งโฟลเดอร์ก่อน"
    )

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

package_lock = json.loads(PACKAGE_LOCK_PATH.read_text(encoding="utf-8"))
package_identity = sha256_file(PACKAGE_LOCK_PATH)
if package_lock.get("packaged_manifest") != PACKAGE_MANIFEST_PATH.name:
    raise RuntimeError("package lock ระบุชื่อ manifest ไม่ตรง")
manifest_sha256 = sha256_file(PACKAGE_MANIFEST_PATH)
if manifest_sha256 != package_lock.get("packaged_manifest_sha256"):
    raise RuntimeError("samples.pretrain.jsonl มี SHA-256 ไม่ตรงกับ package lock")

package_records = [
    json.loads(line)
    for line in PACKAGE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
package_validation = validate_packaged_records(package_records, package_lock)
expected_assets: dict[str, str] = package_validation["expected_assets"]

raw_shards = package_lock.get("shards")
if not isinstance(raw_shards, list) or not raw_shards:
    raise RuntimeError("package lock ขาดรายการ ZIP shards")
verified_shards: list[dict[str, Any]] = []
shard_filenames: set[str] = set()
archive_member_names: set[str] = set()
for shard in raw_shards:
    if not isinstance(shard, dict):
        raise RuntimeError("ข้อมูล ZIP shard ต้องเป็น object")
    filename = canonical_zip_member_name(shard.get("filename"))
    if len(PurePosixPath(filename).parts) != 1 or filename in shard_filenames:
        raise RuntimeError(f"ชื่อ ZIP shard ไม่ปลอดภัยหรือซ้ำ: {filename!r}")
    shard_filenames.add(filename)
    shard_path = SYNFOOT_PACKAGE_DIR / filename
    if not shard_path.is_file():
        raise FileNotFoundError(f"ขาด ZIP shard: {shard_path}")
    expected_size = shard.get("size_bytes")
    expected_digest = shard.get("sha256")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
        raise RuntimeError(f"size_bytes ไม่ถูกต้อง: {filename}")
    if shard_path.stat().st_size != expected_size:
        raise RuntimeError(f"ขนาด ZIP shard ไม่ตรง: {shard_path.name}")
    if not isinstance(expected_digest, str) or sha256_file(shard_path) != expected_digest:
        raise RuntimeError(f"SHA-256 ZIP shard ไม่ตรง: {shard_path.name}")
    verified_shards.append(
        {"filename": filename, "size_bytes": expected_size, "sha256": expected_digest}
    )
    with zipfile.ZipFile(shard_path) as archive:
        for info in archive.infolist():
            name = canonical_zip_member_name(info.filename)
            unix_mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISDIR(unix_mode) or stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"ZIP directories/symlinks are not allowed: {name!r}")
            if name in archive_member_names:
                raise RuntimeError(f"duplicate ZIP member: {name!r}")
            if name not in expected_assets:
                raise RuntimeError(f"unexpected ZIP member: {name!r}")
            archive_member_names.add(name)

missing_assets = set(expected_assets) - archive_member_names
if missing_assets:
    preview = sorted(missing_assets)[:10]
    raise RuntimeError(f"ZIP shards ขาด assets จาก manifest: {preview!r}")
if len(archive_member_names) != len(expected_assets):
    raise RuntimeError("จำนวน ZIP assets ไม่ตรงกับ manifest")

preferred_root = Path(f"/content/synfoot-v2-{package_identity[:16]}")
preferred_marker = preferred_root / ".package-sha256"
if preferred_root.exists() and (
    not preferred_marker.is_file()
    or preferred_marker.read_text(encoding="ascii") != package_identity
):
    SYNFOOT_ROOT = Path(
        tempfile.mkdtemp(prefix=f"synfoot-v2-{package_identity[:16]}-", dir="/content")
    )
else:
    SYNFOOT_ROOT = preferred_root
marker = SYNFOOT_ROOT / ".package-sha256"
is_verified_resume = (
    marker.is_file() and marker.read_text(encoding="ascii") == package_identity
)
if not is_verified_resume:
    SYNFOOT_ROOT.mkdir(parents=True, exist_ok=True)
    extracted_names: set[str] = set()
    for shard in verified_shards:
        shard_path = SYNFOOT_PACKAGE_DIR / shard["filename"]
        with zipfile.ZipFile(shard_path) as archive:
            for info in archive.infolist():
                name = canonical_zip_member_name(info.filename)
                if name in extracted_names:
                    raise RuntimeError(f"duplicate ZIP member: {name!r}")
                extract_verified_zip_member(
                    archive, info, SYNFOOT_ROOT, expected_assets[name]
                )
                extracted_names.add(name)
    if extracted_names != set(expected_assets):
        raise RuntimeError("extracted asset set ไม่ตรงกับ manifest")
    shutil.copy2(PACKAGE_MANIFEST_PATH, SYNFOOT_ROOT / "samples.pretrain.jsonl")
    marker.write_text(package_identity, encoding="ascii")
else:
    for relative_name, expected_digest in expected_assets.items():
        existing = SYNFOOT_ROOT.joinpath(*PurePosixPath(relative_name).parts)
        if (
            not existing.is_file()
            or not existing.resolve().is_relative_to(SYNFOOT_ROOT.resolve())
            or sha256_file(existing) != expected_digest
        ):
            raise RuntimeError(f"extracted asset ไม่ผ่าน SHA-256: {relative_name}")

SYNFOOT_MANIFEST = SYNFOOT_ROOT / "samples.pretrain.jsonl"
if sha256_file(SYNFOOT_MANIFEST) != manifest_sha256:
    raise RuntimeError("manifest ที่แตกแล้วมี SHA-256 ไม่ตรง")
package_provenance = {
    "package_lock_sha256": package_identity,
    "package_lock": package_lock,
    "packaged_manifest": {
        "filename": PACKAGE_MANIFEST_PATH.name,
        "sha256": manifest_sha256,
    },
    "shards": verified_shards,
    "validated_counts": package_validation["counts"],
    "validated_groups": package_validation["groups"],
    "asset_count": len(expected_assets),
}
print("SynFoot พร้อม:", package_validation["counts"], "shards:", len(verified_shards))
""",
            "synfoot-package",
        ),
        code(
            """# ค่าการเทรนและ fingerprint สำหรับ resume จาก Google Drive อย่างปลอดภัย
IMAGE_SIZE = 384
BATCH_SIZE = 4
SYNFOOT_EPOCHS = 25
FOOTNET_ALL_EPOCHS = 35
FOOTNET_DORSAL_EPOCHS = 30
PYTHON = sys.executable

SYNFOOT_RUN_DIR = ARTIFACT_DIR / "synfoot-pretrain"
ALL_VIEWS_DIR = ARTIFACT_DIR / "footnet-all-views"
DORSAL_DIR = ARTIFACT_DIR / "footnet-dorsal"

def run(command: list[str]) -> None:
    print("กำลังทำงาน:", " ".join(map(str, command)))
    subprocess.run(command, cwd=PROJECT_DIR, check=True, env=os.environ.copy())

RUNTIME_REQUIREMENTS = [
    "tensorflow>=2.18,<2.21",
    "numpy>=1.26,<2.2",
    "pillow>=10,<13",
    "requests>=2.31,<3",
]
COMMON_FINGERPRINT_INPUTS = {
    "embedded_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    "synfoot_package_lock_sha256": package_identity,
    "synfoot_package_manifest_sha256": manifest_sha256,
    "synfoot_shards": verified_shards,
    "runtime_requirements": RUNTIME_REQUIREMENTS,
}

def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def make_stage_spec(
    stage: str,
    command: list[str],
    dataset: dict[str, object],
    upstream_checkpoint_sha256: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage": stage,
        "common_inputs": COMMON_FINGERPRINT_INPUTS,
        "dataset": dataset,
        "training_args": command[1:],
        "upstream_checkpoint_sha256": upstream_checkpoint_sha256,
    }

def stage_fingerprint_path(run_dir: Path) -> Path:
    return run_dir / "stage_fingerprint.json"

def completed(run_dir: Path, stage_spec: dict[str, object]) -> bool:
    run_path = run_dir / "run.json"
    checkpoint_path = run_dir / "best.keras"
    fingerprint_path = stage_fingerprint_path(run_dir)
    if not run_path.is_file() or not checkpoint_path.is_file() or not fingerprint_path.is_file():
        return False
    try:
        document = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        document.get("fingerprint") == canonical_json_sha256(stage_spec)
        and document.get("spec") == stage_spec
        and document.get("checkpoint_sha256") == sha256_file(checkpoint_path)
        and document.get("run_sha256") == sha256_file(run_path)
    )

def write_stage_fingerprint(run_dir: Path, stage_spec: dict[str, object]) -> None:
    run_path = run_dir / "run.json"
    checkpoint_path = run_dir / "best.keras"
    if not run_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"training stage ยังไม่สมบูรณ์: {run_dir}")
    document = {
        "schema_version": 1,
        "fingerprint": canonical_json_sha256(stage_spec),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "run_sha256": sha256_file(run_path),
        "spec": stage_spec,
    }
    fingerprint_path = stage_fingerprint_path(run_dir)
    temporary_path = fingerprint_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_path.replace(fingerprint_path)
""",
            "training-config",
        ),
        code(
            """# Stage 1: pretrain ด้วยมุมบน ≤35° และแยก validation ตาม foot-id
SYNFOOT_COMMAND = [
    PYTHON, "ml/train.py",
    "--manifest", str(SYNFOOT_MANIFEST),
    "--split-field", "pretrain_split",
    "--subsets", "measurement-top35",
    "--geometry-augmentation",
    "--image-size", str(IMAGE_SIZE),
    "--batch-size", str(BATCH_SIZE),
    "--epochs", str(SYNFOOT_EPOCHS),
    "--output", str(SYNFOOT_RUN_DIR),
]
SYNFOOT_STAGE_SPEC = make_stage_spec(
    "synfoot-v2-pretrain",
    SYNFOOT_COMMAND,
    {
        "manifest_sha256": manifest_sha256,
        "split_field": "pretrain_split",
        "subset": "measurement-top35",
        "counts": package_validation["counts"],
        "groups": package_validation["groups"],
    },
    None,
)
if completed(SYNFOOT_RUN_DIR, SYNFOOT_STAGE_SPEC):
    print("พบ SynFoot checkpoint แล้ว จึงไม่เทรนซ้ำ")
else:
    stage_fingerprint_path(SYNFOOT_RUN_DIR).unlink(missing_ok=True)
    run(SYNFOOT_COMMAND)
    write_stage_fingerprint(SYNFOOT_RUN_DIR, SYNFOOT_STAGE_SPEC)
""",
            "train-synfoot",
        ),
        code(
            """# ดาวน์โหลด FootNet release ที่ล็อก SHA-256 ไว้
import requests

RECORD_ID = "20457253"
EXPECTED_SHA256 = "3c1bfe4a037f60925325421d467e2310bf65a9c8b2485314a0069b6bbe579b07"
EXPECTED_SIZE = 556_662_306
RAW_DIR = PROJECT_DIR / "ml/data/raw/footnet"
ARCHIVE_PATH = RAW_DIR / "FootNet_v1.0.zip"
RAW_DIR.mkdir(parents=True, exist_ok=True)

valid_existing = (
    ARCHIVE_PATH.exists()
    and ARCHIVE_PATH.stat().st_size == EXPECTED_SIZE
    and sha256_file(ARCHIVE_PATH) == EXPECTED_SHA256
)
if not valid_existing:
    metadata = requests.get(f"https://zenodo.org/api/records/{RECORD_ID}", timeout=60)
    metadata.raise_for_status()
    item = next(item for item in metadata.json()["files"] if item.get("key") == ARCHIVE_PATH.name)
    item_links = item.get("links", {})
    download_url = item_links.get("content") or item_links.get("self")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        raise RuntimeError(f"Zenodo ไม่ได้ส่งลิงก์ดาวน์โหลดที่รองรับ: {item_links!r}")
    partial = ARCHIVE_PATH.with_suffix(".zip.part")
    digest = hashlib.sha256()
    size = 0
    with requests.get(download_url, stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("FootNet ไม่ตรงกับ release ที่ตรวจสอบไว้")
    partial.replace(ARCHIVE_PATH)
print("FootNet ผ่าน SHA-256 แล้ว")
""",
            "footnet-download",
        ),
        code(
            """# Import และสร้าง development holdout ของ FootNet
FOOTNET_ROOT = PROJECT_DIR / "ml/data/processed/footnet"
FOOTNET_MANIFEST = FOOTNET_ROOT / "samples.dev.jsonl"
if not (FOOTNET_ROOT / "dataset.lock.json").is_file():
    run([
        PYTHON, "-m", "ml.footdata", "import", "footnet",
        "--archive", str(ARCHIVE_PATH),
        "--output-root", str(FOOTNET_ROOT),
        "--expected-sha256", EXPECTED_SHA256,
        "--accept-license",
    ])
if not FOOTNET_MANIFEST.is_file():
    run([
        PYTHON, "-m", "ml.footdata", "split",
        str(FOOTNET_ROOT / "samples.jsonl"),
        "--output", str(FOOTNET_MANIFEST),
        "--lock", str(FOOTNET_ROOT / "splits.dev.lock.json"),
        "--allow-unresolved-eval",
    ])
FOOTNET_ARCHIVE_SHA256 = sha256_file(ARCHIVE_PATH)
FOOTNET_MANIFEST_SHA256 = sha256_file(FOOTNET_MANIFEST)
FOOTNET_DATASET_LOCK_PATH = FOOTNET_ROOT / "dataset.lock.json"
FOOTNET_DATASET_LOCK_SHA256 = sha256_file(FOOTNET_DATASET_LOCK_PATH)
print("FootNet manifest พร้อม:", FOOTNET_MANIFEST)
""",
            "footnet-prepare",
        ),
        code(
            """# Stage 2: fine-tune ด้วย FootNet ทุกมุม
ALL_VIEWS_COMMAND = [
    PYTHON, "ml/train.py",
    "--manifest", str(FOOTNET_MANIFEST),
    "--real-only",
    "--initial-model", str(SYNFOOT_RUN_DIR / "best.keras"),
    "--learning-rate", "0.0001",
    "--geometry-augmentation",
    "--image-size", str(IMAGE_SIZE),
    "--batch-size", str(BATCH_SIZE),
    "--epochs", str(FOOTNET_ALL_EPOCHS),
    "--output", str(ALL_VIEWS_DIR),
]
FOOTNET_DATASET_FINGERPRINT = {
    "archive_sha256": FOOTNET_ARCHIVE_SHA256,
    "manifest_sha256": FOOTNET_MANIFEST_SHA256,
    "dataset_lock_sha256": FOOTNET_DATASET_LOCK_SHA256,
    "views": "all",
}
ALL_VIEWS_STAGE_SPEC = make_stage_spec(
    "footnet-all-views",
    ALL_VIEWS_COMMAND,
    FOOTNET_DATASET_FINGERPRINT,
    sha256_file(SYNFOOT_RUN_DIR / "best.keras"),
)
if completed(ALL_VIEWS_DIR, ALL_VIEWS_STAGE_SPEC):
    print("พบ FootNet all-view checkpoint แล้ว")
else:
    stage_fingerprint_path(ALL_VIEWS_DIR).unlink(missing_ok=True)
    run(ALL_VIEWS_COMMAND)
    write_stage_fingerprint(ALL_VIEWS_DIR, ALL_VIEWS_STAGE_SPEC)
""",
            "train-footnet-all",
        ),
        code(
            """# Stage 3: specialize ด้วยภาพ dorsal/top ที่ใกล้กับกล้องในแอป
DORSAL_COMMAND = [
    PYTHON, "ml/train.py",
    "--manifest", str(FOOTNET_MANIFEST),
    "--views", "dorsal",
    "--real-only",
    "--initial-model", str(ALL_VIEWS_DIR / "best.keras"),
    "--learning-rate", "0.00005",
    "--geometry-augmentation",
    "--image-size", str(IMAGE_SIZE),
    "--batch-size", str(BATCH_SIZE),
    "--epochs", str(FOOTNET_DORSAL_EPOCHS),
    "--output", str(DORSAL_DIR),
]
DORSAL_STAGE_SPEC = make_stage_spec(
    "footnet-dorsal",
    DORSAL_COMMAND,
    {**FOOTNET_DATASET_FINGERPRINT, "views": ["dorsal"]},
    sha256_file(ALL_VIEWS_DIR / "best.keras"),
)
if completed(DORSAL_DIR, DORSAL_STAGE_SPEC):
    print("พบ FootNet dorsal checkpoint แล้ว")
else:
    stage_fingerprint_path(DORSAL_DIR).unlink(missing_ok=True)
    run(DORSAL_COMMAND)
    write_stage_fingerprint(DORSAL_DIR, DORSAL_STAGE_SPEC)
""",
            "train-footnet-dorsal",
        ),
        code(
            """# เลือก threshold จาก validation แล้วใช้ค่าเดียวกับ test
VAL_DIR = DORSAL_DIR / "evaluation-val"
TEST_DIR = DORSAL_DIR / "evaluation-test"
run([
    PYTHON, "ml/evaluate.py",
    "--manifest", str(FOOTNET_MANIFEST),
    "--model", str(DORSAL_DIR / "best.keras"),
    "--split", "val",
    "--views", "dorsal",
    "--output", str(VAL_DIR),
])
validation_report = json.loads((VAL_DIR / "summary.json").read_text(encoding="utf-8"))
selected_threshold = float(validation_report["selected"]["threshold"])
run([
    PYTHON, "ml/evaluate.py",
    "--manifest", str(FOOTNET_MANIFEST),
    "--model", str(DORSAL_DIR / "best.keras"),
    "--split", "test",
    "--thresholds", str(selected_threshold),
    "--views", "dorsal",
    "--output", str(TEST_DIR),
])
print("threshold:", selected_threshold)
""",
            "evaluation",
        ),
        code(
            """# Export TensorFlow Lite, deployment contract และ provenance ที่ตรวจย้อนกลับได้
import importlib.metadata
import platform

TFLITE_PATH = ARTIFACT_DIR / "foot_segmenter_synfoot_v2_fp16.tflite"
run([
    PYTHON, "ml/export_tflite.py",
    "--model", str(DORSAL_DIR / "best.keras"),
    "--output", str(TFLITE_PATH),
])
STAGE_SPECS = {
    "synfoot-v2-pretrain": (SYNFOOT_RUN_DIR, SYNFOOT_STAGE_SPEC),
    "footnet-all-views": (ALL_VIEWS_DIR, ALL_VIEWS_STAGE_SPEC),
    "footnet-dorsal": (DORSAL_DIR, DORSAL_STAGE_SPEC),
}
stage_fingerprints: dict[str, object] = {}
for stage_name, (run_dir, stage_spec) in STAGE_SPECS.items():
    if not completed(run_dir, stage_spec):
        raise RuntimeError(f"stage fingerprint/checkpoint ไม่ผ่านก่อน export: {stage_name}")
    stage_fingerprints[stage_name] = json.loads(
        stage_fingerprint_path(run_dir).read_text(encoding="utf-8")
    )

TFLITE_METADATA_PATH = TFLITE_PATH.with_suffix(".json")
provenance = {
    "schema_version": 1,
    "embedded_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    "synfoot_package": package_provenance,
    "footnet": {
        "zenodo_record_id": RECORD_ID,
        "archive": {
            "filename": ARCHIVE_PATH.name,
            "size_bytes": ARCHIVE_PATH.stat().st_size,
            "sha256": FOOTNET_ARCHIVE_SHA256,
        },
        "manifest": {
            "filename": FOOTNET_MANIFEST.name,
            "sha256": FOOTNET_MANIFEST_SHA256,
        },
        "dataset_lock_sha256": FOOTNET_DATASET_LOCK_SHA256,
    },
    "checkpoints": {
        "synfoot-v2-pretrain": sha256_file(SYNFOOT_RUN_DIR / "best.keras"),
        "footnet-all-views": sha256_file(ALL_VIEWS_DIR / "best.keras"),
        "footnet-dorsal": sha256_file(DORSAL_DIR / "best.keras"),
    },
    "stage_fingerprints": stage_fingerprints,
    "evaluation": {
        "validation_summary_sha256": sha256_file(VAL_DIR / "summary.json"),
        "test_summary_sha256": sha256_file(TEST_DIR / "summary.json"),
        "selected_threshold": selected_threshold,
    },
    "export": {
        "tflite_filename": TFLITE_PATH.name,
        "tflite_sha256": sha256_file(TFLITE_PATH),
        "metadata_filename": TFLITE_METADATA_PATH.name,
        "metadata_sha256": sha256_file(TFLITE_METADATA_PATH),
    },
    "environment": {
        "python": platform.python_version(),
        "tensorflow": importlib.metadata.version("tensorflow"),
        "numpy": importlib.metadata.version("numpy"),
        "pillow": importlib.metadata.version("pillow"),
    },
    "validation_warning": (
        "Synthetic and sample-level holdouts do not establish real-person or "
        "millimetre accuracy; phone-camera validation is still required"
    ),
}
PROVENANCE_PATH = ARTIFACT_DIR / "training_provenance.json"
PROVENANCE_PATH.write_text(
    json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
)
deployment = {
    "image_size": IMAGE_SIZE,
    "threshold": selected_threshold,
    "labels": ["background", "foot"],
    "input_normalization": {"mean": [127.5], "std": [127.5]},
    "input_policy": "square_letterbox_then_unletterbox_mask",
    "measurement_tracker_frames": 7,
    "training_stages": ["synfoot-v2-pretrain", "footnet-all-views", "footnet-dorsal"],
    "validation_warning": "FootNet holdouts are sample-level; real participant and millimetre validation is still required",
    "provenance": provenance,
}
DEPLOYMENT_PATH = ARTIFACT_DIR / "deployment_config.json"
DEPLOYMENT_PATH.write_text(
    json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(deployment, ensure_ascii=False, indent=2))
""",
            "export",
        ),
        code(
            """# รวมผลลัพธ์ขนาดเล็กและดาวน์โหลดกลับเครื่อง
from google.colab import files

PACKAGE_DIR = Path("/content/foot-segmentation-synfoot-v2-result")
if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir()
package_files = [
    (TFLITE_PATH, TFLITE_PATH.name),
    (TFLITE_PATH.with_suffix(".json"), TFLITE_PATH.with_suffix(".json").name),
    (DEPLOYMENT_PATH, DEPLOYMENT_PATH.name),
    (PROVENANCE_PATH, PROVENANCE_PATH.name),
    (VAL_DIR / "summary.json", "validation_summary.json"),
    (TEST_DIR / "summary.json", "test_summary.json"),
    (SYNFOOT_RUN_DIR / "run.json", "synfoot_run.json"),
    (SYNFOOT_RUN_DIR / "training.csv", "synfoot_training.csv"),
    (DORSAL_DIR / "run.json", "footnet_dorsal_run.json"),
    (DORSAL_DIR / "training.csv", "footnet_dorsal_training.csv"),
]
for source, destination_name in package_files:
    shutil.copy2(source, PACKAGE_DIR / destination_name)
zip_path = shutil.make_archive(
    "/content/foot-segmentation-synfoot-v2-result", "zip", PACKAGE_DIR
)
files.download(zip_path)
""",
            "download",
        ),
        markdown(
            """## หลังเทรนเสร็จ\n
ส่ง `foot-segmentation-synfoot-v2-result.zip` กลับมาเพื่อตรวจ metric, เปรียบเทียบกับโมเดลเดิม และทดสอบ golden images ก่อนนำไปแทนโมเดลในแอป ห้ามอ้างว่าความแม่นยำดีขึ้นจนกว่าผลเปรียบเทียบและการทดสอบเท้าจริงจะผ่าน\n""",
            "next-step",
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "name": "SynFoot V2 Foot Segmentation Training.ipynb",
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def main() -> None:
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
