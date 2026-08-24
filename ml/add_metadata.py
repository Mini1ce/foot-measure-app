"""Attach MediaPipe Image Segmenter metadata and labels to a TFLite model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path(__file__).parent / "labels.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from tflite_support.metadata_writers import image_segmenter, writer_utils
    except ImportError as error:
        raise SystemExit(
            "tflite-support is missing. Its 0.4.4 wheel is not available on native "
            "Windows; use Python 3.10/3.11 in WSL/Linux and install "
            "ml/requirements-metadata.txt"
        ) from error

    writer = image_segmenter.MetadataWriter.create_for_inference(
        writer_utils.load_file(str(args.model)),
        [127.5],
        [127.5],
        [str(args.labels)],
    )
    populated = writer.populate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer_utils.save_file(populated, str(args.output))
    metadata_json = json.loads(writer.get_metadata_json())
    report = {
        "model": str(args.output.resolve()),
        "sha256": hashlib.sha256(populated).hexdigest(),
        "bytes": len(populated),
        "labels": args.labels.read_text(encoding="utf-8").splitlines(),
        "metadata": metadata_json,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Metadata model written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
