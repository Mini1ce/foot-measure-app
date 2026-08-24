"""Build the self-contained Google Colab notebook for foot segmentation training."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("Foot_Segmentation_Training.ipynb")


def source_files() -> list[Path]:
    paths = [
        Path("ml/train.py"),
        Path("ml/evaluate.py"),
        Path("ml/export_tflite.py"),
        Path("ml/labels.txt"),
        Path("ml/requirements-train.txt"),
    ]
    paths.extend(
        path.relative_to(PROJECT_ROOT)
        for folder in (PROJECT_ROOT / "ml" / "footseg", PROJECT_ROOT / "ml" / "footdata")
        for path in sorted(folder.glob("*.py"))
    )
    return paths


def embedded_source_archive() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_path in source_files():
            absolute_path = PROJECT_ROOT / relative_path
            info = zipfile.ZipInfo(relative_path.as_posix(), date_time=(2026, 8, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, absolute_path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def markdown(text: str, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build_notebook() -> dict[str, object]:
    source_archive = embedded_source_archive()
    cells = [
        markdown(
            """# เทรนโมเดลตรวจจับเท้าสำหรับ Foot Measure\n
สมุดนี้ทำงานให้อัตโนมัติตั้งแต่ดาวน์โหลด FootNet, ตรวจสอบไฟล์, เทรนโมเดลสองช่วง, ประเมินผล, เลือก threshold และแปลงเป็น TensorFlow Lite\n
ก่อนเริ่มให้เลือก **Runtime → Change runtime type → GPU** แล้วกด **Runtime → Run all** ระบบจะขอเชื่อมต่อ Google Drive หนึ่งครั้งเพื่อเก็บ checkpoint และโมเดลไม่ให้หายเมื่อ Colab ตัดการเชื่อมต่อ\n
Dataset FootNet ใช้สัญญาอนุญาต [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) และต้องคงเครดิตผู้จัดทำ: Adith Vijay, Akash Prabhune, Vinay R Srihari และ Akhila Rayampalli\n
> หมายเหตุ: ชุดข้อมูลนี้ช่วยเรื่องขอบเขตเท้า แต่ยังไม่ใช่หลักฐานความแม่นยำระดับมิลลิเมตร การวัดจริงยังต้องทดสอบกับภาพจากแอป บัตรอ้างอิง และขนาดเท้าที่วัดด้วยเครื่องมือจริง\n""",
            "intro",
        ),
        code(
            """# ตรวจว่า Colab ได้เปิด GPU แล้ว
import os
import subprocess

gpu_check = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if gpu_check.returncode != 0:
    raise RuntimeError("ยังไม่ได้เปิด GPU: ไปที่ Runtime > Change runtime type > GPU แล้ว Run all ใหม่")
print("GPU พร้อม:", gpu_check.stdout.strip())
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
""",
            "gpu-check",
        ),
        code(
            """# ติดตั้งรุ่นที่ทดสอบร่วมกับชุดเทรนนี้
import sys
import subprocess

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
print("ติดตั้งส่วนที่จำเป็นเรียบร้อย")
""",
            "dependencies",
        ),
        code(
            f'''# แตกชุดโค้ดที่ฝังมากับสมุดนี้ ไม่ต้องอัปโหลดโปรเจกต์เพิ่ม
import base64
import io
import zipfile
from pathlib import Path

PROJECT_DIR = Path("/content/foot-measure-app")
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
SOURCE_ARCHIVE_B64 = "{source_archive}"
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_ARCHIVE_B64))) as archive:
    archive.extractall(PROJECT_DIR)
print("เตรียมชุดเทรนแล้ว:", PROJECT_DIR)
''',
            "embedded-code",
        ),
        code(
            """# เชื่อม Google Drive เพื่อเก็บ checkpoint และผลลัพธ์
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")
ARTIFACT_DIR = Path("/content/drive/MyDrive/foot-measure-training")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
print("ผลลัพธ์จะถูกเก็บที่:", ARTIFACT_DIR)
""",
            "drive",
        ),
        code(
            """# ดาวน์โหลด FootNet จาก Zenodo record ที่ล็อกรุ่นไว้ และตรวจ SHA-256
import hashlib
import requests

RECORD_ID = "20457253"
EXPECTED_SHA256 = "3c1bfe4a037f60925325421d467e2310bf65a9c8b2485314a0069b6bbe579b07"
EXPECTED_SIZE = 556_662_306
RAW_DIR = PROJECT_DIR / "ml/data/raw/footnet"
ARCHIVE_PATH = RAW_DIR / "FootNet_v1.0.zip"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

valid_existing = (
    ARCHIVE_PATH.exists()
    and ARCHIVE_PATH.stat().st_size == EXPECTED_SIZE
    and sha256_file(ARCHIVE_PATH) == EXPECTED_SHA256
)
if not valid_existing:
    metadata = requests.get(f"https://zenodo.org/api/records/{RECORD_ID}", timeout=60)
    metadata.raise_for_status()
    files = metadata.json()["files"]
    item = next(item for item in files if item.get("key") == ARCHIVE_PATH.name)
    # Zenodo's current record API exposes the binary URL as links.self,
    # while older responses used links.content. Accept both shapes.
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
                print(f"ดาวน์โหลดแล้ว {size / 1024**2:.0f} / {EXPECTED_SIZE / 1024**2:.0f} MiB", end="\\r")
    if size != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise RuntimeError("ไฟล์ FootNet ไม่ตรงกับรุ่นที่ตรวจสอบไว้ จึงหยุดเพื่อความปลอดภัย")
    partial.replace(ARCHIVE_PATH)

print("FootNet ผ่านการตรวจสอบ:", ARCHIVE_PATH)
""",
            "dataset-download",
        ),
        code(
            """# สร้าง manifest และชุดแบ่ง train/validation/test สำหรับการพัฒนา
import json

def run(command: list[str]) -> None:
    print("กำลังทำงาน:", " ".join(map(str, command)))
    subprocess.run(command, cwd=PROJECT_DIR, check=True, env=os.environ.copy())

PYTHON = sys.executable
DATA_ROOT = PROJECT_DIR / "ml/data/processed/footnet"
MANIFEST = DATA_ROOT / "samples.dev.jsonl"

if not (DATA_ROOT / "dataset.lock.json").exists():
    run([
        PYTHON, "-m", "ml.footdata", "import", "footnet",
        "--archive", str(ARCHIVE_PATH),
        "--output-root", str(DATA_ROOT),
        "--expected-sha256", EXPECTED_SHA256,
        "--accept-license",
    ])
if not MANIFEST.exists():
    run([
        PYTHON, "-m", "ml.footdata", "split",
        str(DATA_ROOT / "samples.jsonl"),
        "--output", str(MANIFEST),
        "--lock", str(DATA_ROOT / "splits.dev.lock.json"),
        "--allow-unresolved-eval",
    ])

counts = {"train": 0, "val": 0, "test": 0}
views = {}
for line in MANIFEST.read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    counts[record["split"]] += 1
    views[record["view"]] = views.get(record["view"], 0) + 1
print("แบ่งข้อมูล:", counts, "มุมภาพ:", views)
""",
            "dataset-prepare",
        ),
        code(
            """# ค่าหลัก ปรับ BATCH_SIZE เป็น 2 ได้ถ้า GPU หน่วยความจำไม่พอ
IMAGE_SIZE = 384
BATCH_SIZE = 4
ALL_VIEW_EPOCHS = 45
DORSAL_EPOCHS = 35
ALL_VIEWS_DIR = ARTIFACT_DIR / "footnet-all-views"
DORSAL_DIR = ARTIFACT_DIR / "footnet-dorsal"

def completed(run_dir: Path) -> bool:
    return (run_dir / "run.json").exists() and (run_dir / "best.keras").exists()

if completed(ALL_VIEWS_DIR):
    print("พบโมเดลรอบทุกมุมใน Drive แล้ว จึงไม่เทรนซ้ำ")
else:
    run([
        PYTHON, "ml/train.py",
        "--manifest", str(MANIFEST),
        "--real-only",
        "--image-size", str(IMAGE_SIZE),
        "--batch-size", str(BATCH_SIZE),
        "--epochs", str(ALL_VIEW_EPOCHS),
        "--output", str(ALL_VIEWS_DIR),
    ])
""",
            "train-all",
        ),
        code(
            """# รอบที่สอง: ปรับโมเดลให้เชี่ยวชาญภาพมุมบนเท้าที่แอปใช้งาน
if completed(DORSAL_DIR):
    print("พบโมเดลรอบมุมบนใน Drive แล้ว จึงไม่เทรนซ้ำ")
else:
    run([
        PYTHON, "ml/train.py",
        "--manifest", str(MANIFEST),
        "--views", "dorsal",
        "--real-only",
        "--initial-model", str(ALL_VIEWS_DIR / "best.keras"),
        "--learning-rate", "0.0001",
        "--image-size", str(IMAGE_SIZE),
        "--batch-size", str(BATCH_SIZE),
        "--epochs", str(DORSAL_EPOCHS),
        "--output", str(DORSAL_DIR),
    ])
""",
            "train-dorsal",
        ),
        code(
            """# เลือก threshold จาก validation เท่านั้น แล้วใช้ค่าคงที่เดียวกันกับ test
VAL_DIR = DORSAL_DIR / "evaluation-val"
TEST_DIR = DORSAL_DIR / "evaluation-test"
run([
    PYTHON, "ml/evaluate.py",
    "--manifest", str(MANIFEST),
    "--model", str(DORSAL_DIR / "best.keras"),
    "--split", "val",
    "--views", "dorsal",
    "--output", str(VAL_DIR),
])
validation_report = json.loads((VAL_DIR / "summary.json").read_text(encoding="utf-8"))
selected_threshold = float(validation_report["selected"]["threshold"])
print("threshold ที่เลือกจาก validation:", selected_threshold)

run([
    PYTHON, "ml/evaluate.py",
    "--manifest", str(MANIFEST),
    "--model", str(DORSAL_DIR / "best.keras"),
    "--split", "test",
    "--thresholds", str(selected_threshold),
    "--views", "dorsal",
    "--output", str(TEST_DIR),
])
""",
            "evaluation",
        ),
        code(
            """# แปลงเป็น TensorFlow Lite และสร้างค่าตั้งต้นสำหรับนำไปใส่แอป
TFLITE_PATH = ARTIFACT_DIR / "foot_segmenter_fp16.tflite"
run([
    PYTHON, "ml/export_tflite.py",
    "--model", str(DORSAL_DIR / "best.keras"),
    "--output", str(TFLITE_PATH),
])

deployment = {
    "image_size": IMAGE_SIZE,
    "threshold": selected_threshold,
    "input_normalization": {"mean": [127.5], "std": [127.5]},
    "labels": ["background", "foot"],
    "input_policy": "square_letterbox_then_unletterbox_mask",
    "measurement_tracker_frames": 7,
    "validation_warning": "FootNet holdouts are sample-level because public subject IDs are unavailable",
}
DEPLOYMENT_PATH = ARTIFACT_DIR / "deployment_config.json"
DEPLOYMENT_PATH.write_text(json.dumps(deployment, indent=2), encoding="utf-8")
print(json.dumps(deployment, indent=2))
""",
            "export",
        ),
        code(
            """# รวมไฟล์ที่ต้องใช้เป็น ZIP และดาวน์โหลดกลับเครื่อง
import shutil
from google.colab import files

PACKAGE_DIR = Path("/content/foot-segmentation-result")
if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir()
package_files = [
    (TFLITE_PATH, TFLITE_PATH.name),
    (TFLITE_PATH.with_suffix(".json"), TFLITE_PATH.with_suffix(".json").name),
    (DEPLOYMENT_PATH, DEPLOYMENT_PATH.name),
    (VAL_DIR / "summary.json", "validation_summary.json"),
    (TEST_DIR / "summary.json", "test_summary.json"),
    (DORSAL_DIR / "training.csv", "training.csv"),
    (DORSAL_DIR / "run.json", "run.json"),
]
for source, destination_name in package_files:
    shutil.copy2(source, PACKAGE_DIR / destination_name)

zip_path = shutil.make_archive("/content/foot-segmentation-result", "zip", PACKAGE_DIR)
print("เสร็จแล้ว โมเดลฉบับเต็มยังอยู่ใน Google Drive ที่", ARTIFACT_DIR)
files.download(zip_path)
""",
            "download",
        ),
        markdown(
            """## หลังจากได้ไฟล์ ZIP\n
เก็บไฟล์ `foot-segmentation-result.zip` ไว้ แล้วส่งกลับมาในแชตนี้ ผมจะตรวจผลการเทรน เพิ่ม metadata สำหรับ MediaPipe ทดสอบโมเดลกับไฟล์ตัวอย่าง และเชื่อมเข้ากับระบบวัดเท้า 7 เฟรมต่อให้ครับ\n""",
            "next-step",
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "name": "Foot Segmentation Training.ipynb",
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
    # Windows may expose a legacy console encoding even when the project path
    # contains Thai characters. Keep notebook generation successful there.
    print("Wrote ml/colab/Foot_Segmentation_Training.ipynb")


if __name__ == "__main__":
    main()
