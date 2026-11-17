"""Build the audit/evaluate/fine-tune continuation notebook."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from .build_notebook import code, embedded_source_archive, markdown
from .build_synfoot_notebook import (
    canonical_zip_member_name,
    extract_verified_zip_member,
)


OUTPUT = Path(__file__).with_name("SynFoot_V2_Resume_FineTune.ipynb")
SAFE_ZIP_HELPERS_SOURCE = "\n\n".join(
    inspect.getsource(function)
    for function in (canonical_zip_member_name, extract_verified_zip_member)
)


def build_notebook() -> dict[str, object]:
    source_archive = embedded_source_archive()
    cells = [
        markdown(
            """# ตรวจ mask, ประเมิน และ fine-tune ต่อจาก checkpoint เดิม

Notebook นี้ **ไม่เทรน SynFoot ใหม่** และไม่เขียนทับ `synfoot-pretrain`, `footnet-all-views` หรือ `footnet-dorsal` เดิม ระบบจะตรวจ SHA-256/fingerprint ของ checkpoint ทั้งสามช่วงก่อน จากนั้นสร้างรายงานภาพสำหรับตรวจ FootNet dorsal masks

ค่าเริ่มต้น `FINE_TUNE_EPOCHS = 0` หมายถึงประเมิน checkpoint เดิมอย่างเดียว หลังตรวจทุกรูปในรายงานและวาง `measurement-review.csv` แล้ว จึงเปลี่ยนเป็นจำนวน epoch ที่ต้องการ เช่น 15

> FootNet split นี้เป็น development/sample-level holdout และ test ถูกเปิดดูแล้ว ผลจึงใช้วินิจฉัยเท่านั้น ไม่ใช่หลักฐานความแม่นยำกับคนใหม่หรือหน่วยมิลลิเมตร
""",
            "intro",
        ),
        code(
            """# เปิด T4 GPU
import os
import subprocess

gpu_check = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if gpu_check.returncode != 0:
    raise RuntimeError("เลือก Runtime > Change runtime type > T4 GPU ก่อน")
print("GPU พร้อม:", gpu_check.stdout.strip())
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
""",
            "gpu-check",
        ),
        code(
            """# ติดตั้ง dependencies
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
""",
            "dependencies",
        ),
        code(
            f'''# แตก source ที่ฝังใน notebook พร้อมตรวจ path
import base64
import hashlib
import io
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

{SAFE_ZIP_HELPERS_SOURCE}

SOURCE_ARCHIVE_B64 = "{source_archive}"
SOURCE_ARCHIVE_BYTES = base64.b64decode(SOURCE_ARCHIVE_B64, validate=True)
SOURCE_ARCHIVE_SHA256 = hashlib.sha256(SOURCE_ARCHIVE_BYTES).hexdigest()
PROJECT_DIR = Path(f"/content/foot-measure-resume-{{SOURCE_ARCHIVE_SHA256[:16]}}")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(SOURCE_ARCHIVE_BYTES)) as archive:
    seen: set[str] = set()
    for info in archive.infolist():
        name = canonical_zip_member_name(info.filename)
        if name in seen:
            raise RuntimeError(f"duplicate embedded source: {{name}}")
        seen.add(name)
        destination = PROJECT_DIR.joinpath(*PurePosixPath(name).parts)
        if destination.exists():
            if destination.is_file() and hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(archive.read(info)).digest():
                continue
            raise RuntimeError(f"embedded source conflict: {{name}}")
        extract_verified_zip_member(archive, info, PROJECT_DIR, None)
print("source SHA-256:", SOURCE_ARCHIVE_SHA256)
''',
            "embedded-code",
        ),
        code(
            """# เชื่อม Drive และตั้งค่า
from google.colab import drive

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/foot-measure-training")
BASE_RUN_DIR = DRIVE_ROOT / "synfoot-v2-run"
BASE_DORSAL_DIR = BASE_RUN_DIR / "footnet-dorsal"
REVIEW_DIR = DRIVE_ROOT / "footnet-measurement-review"
REVIEW_CSV = REVIEW_DIR / "measurement-review.csv"
REFINED_DIR = BASE_RUN_DIR / "footnet-dorsal-refined"

IMAGE_SIZE = 384
BATCH_SIZE = 4
FINE_TUNE_EPOCHS = 0  # เปลี่ยนเป็น 10-20 หลังตรวจ mask ครบเท่านั้น
FINE_TUNE_LEARNING_RATE = 0.00001
PYTHON = sys.executable

def run(command: list[str]) -> None:
    print("กำลังทำงาน:", " ".join(map(str, command)))
    subprocess.run(command, cwd=PROJECT_DIR, check=True, env=os.environ.copy())

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
""",
            "drive-config",
        ),
        code(
            """# ตรวจ checkpoint/fingerprint เดิมโดยไม่ผูกกับ source notebook รุ่นใหม่
STAGE_DIRS = {
    "synfoot-v2-pretrain": BASE_RUN_DIR / "synfoot-pretrain",
    "footnet-all-views": BASE_RUN_DIR / "footnet-all-views",
    "footnet-dorsal": BASE_DORSAL_DIR,
}

def verify_saved_stage(stage_name: str, run_dir: Path) -> dict[str, object]:
    checkpoint = run_dir / "best.keras"
    run_json = run_dir / "run.json"
    fingerprint_path = run_dir / "stage_fingerprint.json"
    for path in (checkpoint, run_json, fingerprint_path):
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint เดิมไม่ครบ: {path}")
    document = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    spec = document.get("spec")
    if not isinstance(spec, dict) or spec.get("stage") != stage_name:
        raise RuntimeError(f"stage spec ไม่ตรง: {stage_name}")
    if document.get("fingerprint") != canonical_json_sha256(spec):
        raise RuntimeError(f"stage fingerprint เสีย: {stage_name}")
    if document.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError(f"checkpoint SHA-256 ไม่ตรง: {stage_name}")
    if document.get("run_sha256") != sha256_file(run_json):
        raise RuntimeError(f"run.json SHA-256 ไม่ตรง: {stage_name}")
    return document

verified_stages = {
    name: verify_saved_stage(name, directory)
    for name, directory in STAGE_DIRS.items()
}
synfoot_hash = verified_stages["synfoot-v2-pretrain"]["checkpoint_sha256"]
all_views_hash = verified_stages["footnet-all-views"]["checkpoint_sha256"]
dorsal_hash = verified_stages["footnet-dorsal"]["checkpoint_sha256"]
if verified_stages["footnet-all-views"]["spec"].get("upstream_checkpoint_sha256") != synfoot_hash:
    raise RuntimeError("สาย checkpoint SynFoot → all views ไม่ตรง")
if verified_stages["footnet-dorsal"]["spec"].get("upstream_checkpoint_sha256") != all_views_hash:
    raise RuntimeError("สาย checkpoint all views → dorsal ไม่ตรง")
print("checkpoint เดิมผ่าน:", dorsal_hash)
""",
            "verify-base-checkpoints",
        ),
        code(
            """# ดาวน์โหลด FootNet release แบบล็อกขนาดและ SHA-256
import requests

RECORD_ID = "20457253"
EXPECTED_SHA256 = "3c1bfe4a037f60925325421d467e2310bf65a9c8b2485314a0069b6bbe579b07"
EXPECTED_SIZE = 556_662_306
RAW_DIR = PROJECT_DIR / "ml/data/raw/footnet"
ARCHIVE_PATH = RAW_DIR / "FootNet_v1.0.zip"
RAW_DIR.mkdir(parents=True, exist_ok=True)

valid_existing = ARCHIVE_PATH.is_file() and ARCHIVE_PATH.stat().st_size == EXPECTED_SIZE and sha256_file(ARCHIVE_PATH) == EXPECTED_SHA256
if not valid_existing:
    metadata = requests.get(f"https://zenodo.org/api/records/{RECORD_ID}", timeout=60)
    metadata.raise_for_status()
    item = next(item for item in metadata.json()["files"] if item.get("key") == ARCHIVE_PATH.name)
    links = item.get("links", {})
    download_url = links.get("content") or links.get("self")
    if not isinstance(download_url, str) or not download_url.startswith("https://"):
        raise RuntimeError(f"Zenodo ไม่มีลิงก์ที่รองรับ: {links!r}")
    partial = ARCHIVE_PATH.with_suffix(".zip.part")
    digest = hashlib.sha256()
    size = 0
    with requests.get(download_url, stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("FootNet ไม่ตรง release ที่ล็อกไว้")
    partial.replace(ARCHIVE_PATH)
print("FootNet archive ผ่าน")
""",
            "footnet-download",
        ),
        code(
            """# Import/split แบบ deterministic และสร้างรายงาน mask บน Drive
FOOTNET_ROOT = PROJECT_DIR / "ml/data/processed/footnet"
FOOTNET_MANIFEST = FOOTNET_ROOT / "samples.dev.jsonl"
if not (FOOTNET_ROOT / "dataset.lock.json").is_file():
    run([
        PYTHON, "-m", "ml.footdata", "import", "footnet",
        "--archive", str(ARCHIVE_PATH), "--output-root", str(FOOTNET_ROOT),
        "--expected-sha256", EXPECTED_SHA256, "--accept-license",
    ])
if not FOOTNET_MANIFEST.is_file():
    run([
        PYTHON, "-m", "ml.footdata", "split", str(FOOTNET_ROOT / "samples.jsonl"),
        "--output", str(FOOTNET_MANIFEST),
        "--lock", str(FOOTNET_ROOT / "splits.dev.lock.json"),
        "--allow-unresolved-eval",
    ])
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
run([
    PYTHON, "-m", "ml.footdata", "measurement-audit", str(FOOTNET_MANIFEST),
    "--data-root", str(FOOTNET_ROOT), "--output-dir", str(REVIEW_DIR),
    "--views", "dorsal",
])
from google.colab import files
report_zip = shutil.make_archive("/content/footnet-measurement-review", "zip", REVIEW_DIR)
files.download(report_zip)
print("แตก ZIP ที่ดาวน์โหลดแล้วเปิด index.html ในเครื่อง")
print("ตรวจทุกรูป แล้วดาวน์โหลด CSV จากรายงานกลับมาวางเป็น:", REVIEW_CSV)
""",
            "measurement-audit",
        ),
        code(
            """# ใช้เฉพาะการตัดสินใจ include/exclude ที่กรอกครบ
REVIEWED_MANIFEST = FOOTNET_ROOT / "samples.measurement-reviewed.jsonl"
REVIEWED_LOCK = FOOTNET_ROOT / "samples.measurement-reviewed.lock.json"
if REVIEW_CSV.is_file():
    run([
        PYTHON, "-m", "ml.footdata", "measurement-filter", str(FOOTNET_MANIFEST),
        "--review", str(REVIEW_CSV), "--output", str(REVIEWED_MANIFEST),
        "--lock", str(REVIEWED_LOCK),
    ])
    ACTIVE_MANIFEST = REVIEWED_MANIFEST
    print("ใช้ reviewed manifest:", ACTIVE_MANIFEST)
else:
    ACTIVE_MANIFEST = FOOTNET_MANIFEST
    print("ยังไม่มี measurement-review.csv: ประเมินแบบ diagnostic เท่านั้น; fine-tune ถูกห้าม")

active_records = [
    json.loads(line)
    for line in ACTIVE_MANIFEST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
active_counts = {
    split: sum(record.get("split") == split and record.get("view") == "dorsal" for record in active_records)
    for split in ("train", "val", "test")
}
if active_counts["val"] == 0:
    raise RuntimeError("manifest ต้องมี dorsal validation อย่างน้อยหนึ่งภาพ")
print("dorsal counts:", active_counts)
""",
            "review-filter",
        ),
        code(
            """# Fine-tune ต่อจาก footnet-dorsal/best.keras; ค่า 0 จะไม่เทรน
BASE_CHECKPOINT = BASE_DORSAL_DIR / "best.keras"
CANDIDATE_CHECKPOINT = BASE_CHECKPOINT
REFINED_FINGERPRINT = REFINED_DIR / "stage_fingerprint.json"

if FINE_TUNE_EPOCHS < 0:
    raise ValueError("FINE_TUNE_EPOCHS ต้องไม่ติดลบ")
if FINE_TUNE_EPOCHS > 0:
    if not REVIEW_CSV.is_file() or ACTIVE_MANIFEST != REVIEWED_MANIFEST:
        raise RuntimeError("ห้าม fine-tune ก่อนกรอก measurement-review.csv ครบ")
    command = [
        PYTHON, "ml/train.py", "--manifest", str(REVIEWED_MANIFEST),
        "--views", "dorsal", "--real-only", "--initial-model", str(BASE_CHECKPOINT),
        "--learning-rate", str(FINE_TUNE_LEARNING_RATE), "--geometry-augmentation",
        "--image-size", str(IMAGE_SIZE), "--batch-size", str(BATCH_SIZE),
        "--epochs", str(FINE_TUNE_EPOCHS), "--output", str(REFINED_DIR),
    ]
    refined_spec = {
        "schema_version": 1,
        "stage": "footnet-dorsal-measurement-reviewed",
        "embedded_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "base_checkpoint_sha256": dorsal_hash,
        "reviewed_manifest_sha256": sha256_file(REVIEWED_MANIFEST),
        "review_lock_sha256": sha256_file(REVIEWED_LOCK),
        "training_args": command[1:],
    }
    refined_fingerprint_value = canonical_json_sha256(refined_spec)
    if REFINED_FINGERPRINT.is_file():
        document = json.loads(REFINED_FINGERPRINT.read_text(encoding="utf-8"))
        if document.get("fingerprint") != refined_fingerprint_value or document.get("spec") != refined_spec:
            raise RuntimeError("refined output มี fingerprint คนละชุด; เปลี่ยนชื่อโฟลเดอร์ก่อน")
        if document.get("checkpoint_sha256") != sha256_file(REFINED_DIR / "best.keras"):
            raise RuntimeError("refined checkpoint SHA-256 ไม่ตรง")
        print("พบ refined checkpoint เดิม จึงไม่เทรนซ้ำ")
    else:
        if REFINED_DIR.exists() and any(REFINED_DIR.iterdir()):
            raise RuntimeError("refined output มีไฟล์แต่ไม่มี fingerprint; ไม่เขียนทับ")
        run(command)
        document = {
            "schema_version": 1,
            "fingerprint": refined_fingerprint_value,
            "checkpoint_sha256": sha256_file(REFINED_DIR / "best.keras"),
            "run_sha256": sha256_file(REFINED_DIR / "run.json"),
            "spec": refined_spec,
        }
        temporary = REFINED_FINGERPRINT.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(REFINED_FINGERPRINT)
    CANDIDATE_CHECKPOINT = REFINED_DIR / "best.keras"
else:
    print("FINE_TUNE_EPOCHS=0: ใช้ checkpoint เดิมเพื่อประเมิน")
""",
            "fine-tune",
        ),
        code(
            """# เลือก threshold จาก validation; test ใช้ threshold เดียวและเป็น diagnostic
EVALUATION_ROOT = (REFINED_DIR if FINE_TUNE_EPOCHS > 0 else BASE_DORSAL_DIR) / "resume-evaluation"
VAL_DIR = EVALUATION_ROOT / "validation"
TEST_DIR = EVALUATION_ROOT / "test"
run([
    PYTHON, "ml/evaluate.py", "--manifest", str(ACTIVE_MANIFEST),
    "--model", str(CANDIDATE_CHECKPOINT), "--split", "val", "--views", "dorsal",
    "--output", str(VAL_DIR),
])
validation_report = json.loads((VAL_DIR / "summary.json").read_text(encoding="utf-8"))
selected_threshold = float(validation_report["selected"]["threshold"])
if active_counts["test"]:
    run([
        PYTHON, "ml/evaluate.py", "--manifest", str(ACTIVE_MANIFEST),
        "--model", str(CANDIDATE_CHECKPOINT), "--split", "test",
        "--thresholds", str(selected_threshold), "--views", "dorsal",
        "--output", str(TEST_DIR),
    ])
else:
    print("ไม่มี reviewed test: ข้าม test evaluation")
print("threshold:", selected_threshold)
""",
            "evaluation",
        ),
        code(
            """# Export candidate และรวมรายงานขนาดเล็ก
from google.colab import files

EXPORT_DIR = EVALUATION_ROOT / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
TFLITE_PATH = EXPORT_DIR / "foot_segmenter_candidate_fp16.tflite"
run([
    PYTHON, "ml/export_tflite.py", "--model", str(CANDIDATE_CHECKPOINT),
    "--output", str(TFLITE_PATH),
])
PROVENANCE = {
    "schema_version": 1,
    "mode": "fine-tune" if FINE_TUNE_EPOCHS > 0 else "evaluation-only",
    "base_checkpoint_sha256": dorsal_hash,
    "candidate_checkpoint_sha256": sha256_file(CANDIDATE_CHECKPOINT),
    "embedded_source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    "active_manifest_sha256": sha256_file(ACTIVE_MANIFEST),
    "review_csv_sha256": sha256_file(REVIEW_CSV) if REVIEW_CSV.is_file() else None,
    "selected_threshold": selected_threshold,
    "validation_summary_sha256": sha256_file(VAL_DIR / "summary.json"),
    "test_summary_sha256": sha256_file(TEST_DIR / "summary.json") if (TEST_DIR / "summary.json").is_file() else None,
    "warning": "Diagnostic sample-level evaluation; real phone, reference-card, and millimetre validation remains required",
}
(EXPORT_DIR / "resume_provenance.json").write_text(
    json.dumps(PROVENANCE, ensure_ascii=False, indent=2), encoding="utf-8"
)
PACKAGE_DIR = Path("/content/foot-segmentation-resume-result")
if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir()
for source, name in (
    (TFLITE_PATH, TFLITE_PATH.name),
    (TFLITE_PATH.with_suffix(".json"), TFLITE_PATH.with_suffix(".json").name),
    (EXPORT_DIR / "resume_provenance.json", "resume_provenance.json"),
    (VAL_DIR / "summary.json", "validation_summary.json"),
    (VAL_DIR / "per_sample.csv", "validation_per_sample.csv"),
):
    shutil.copy2(source, PACKAGE_DIR / name)
if (TEST_DIR / "summary.json").is_file():
    shutil.copy2(TEST_DIR / "summary.json", PACKAGE_DIR / "test_summary.json")
    shutil.copy2(TEST_DIR / "per_sample.csv", PACKAGE_DIR / "test_per_sample.csv")
zip_path = shutil.make_archive("/content/foot-segmentation-resume-result", "zip", PACKAGE_DIR)
files.download(zip_path)
""",
            "export-download",
        ),
        markdown(
            """## วิธีใช้สั้น ๆ

1. Run all ครั้งแรกโดยคง `FINE_TUNE_EPOCHS = 0`
2. แตก `footnet-measurement-review.zip` ที่ดาวน์โหลด แล้วเปิด `index.html` ตรวจภาพซ้าย/ขวาทุกรูป เลือก include/exclude และดาวน์โหลด CSV
3. วาง CSV เป็น `footnet-measurement-review/measurement-review.csv`
4. รันใหม่เพื่อสร้าง reviewed manifest และประเมินฐาน
5. หากรายการถูกต้อง เปลี่ยน `FINE_TUNE_EPOCHS` เป็น 10–20 แล้วรันใหม่

อย่านำ TFLite ไปแทนในแอปจนกว่าจะเปรียบเทียบ MAE ความยาว/ความกว้างด้วยโทรศัพท์ บัตร ID-1 และเท้าจริง
""",
            "instructions",
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "name": "SynFoot V2 Resume FineTune.ipynb",
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
