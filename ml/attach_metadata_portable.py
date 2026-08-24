"""Attach MediaPipe Image Segmenter metadata without native support binaries.

This uses the Apache-2.0 generated FlatBuffer schemas shipped in Google's
``tflite-support==0.4.4`` wheel. It is useful on native Windows, where that
wheel's compiled metadata helpers are not available.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path


METADATA_FIELD_NAME = "TFLITE_METADATA"
TFLITE_IDENTIFIER = b"TFL3"
METADATA_IDENTIFIER = b"M001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path, default=Path(__file__).parent / "labels.txt"
    )
    return parser.parse_args()


def _dependencies():
    try:
        import flatbuffers
        from tensorflow_lite_support.metadata import (
            metadata_schema_py_generated as metadata_schema,
        )
        from tensorflow_lite_support.metadata import schema_py_generated as model_schema
    except ImportError as error:
        raise SystemExit(
            "Missing pure FlatBuffer schema dependencies. Install flatbuffers and "
            "make tensorflow_lite_support.metadata generated schemas importable."
        ) from error
    return flatbuffers, metadata_schema, model_schema


def _metadata_buffer(flatbuffers, metadata_schema, label_name: str) -> bytes:
    input_stats = metadata_schema.StatsT()
    input_stats.min = [-1.0]
    input_stats.max = [1.0]

    input_properties = metadata_schema.ImagePropertiesT()
    input_properties.colorSpace = metadata_schema.ColorSpaceType.RGB
    input_content = metadata_schema.ContentT()
    input_content.contentPropertiesType = metadata_schema.ContentProperties.ImageProperties
    input_content.contentProperties = input_properties

    normalization_options = metadata_schema.NormalizationOptionsT()
    normalization_options.mean = [127.5]
    normalization_options.std = [127.5]
    normalization = metadata_schema.ProcessUnitT()
    normalization.optionsType = metadata_schema.ProcessUnitOptions.NormalizationOptions
    normalization.options = normalization_options

    input_metadata = metadata_schema.TensorMetadataT()
    input_metadata.name = "image"
    input_metadata.description = "Square RGB foot image with aspect-ratio-preserving letterbox padding."
    input_metadata.stats = input_stats
    input_metadata.content = input_content
    input_metadata.processUnits = [normalization]

    output_stats = metadata_schema.StatsT()
    output_stats.min = [0.0]
    output_stats.max = [1.0]

    output_properties = metadata_schema.ImagePropertiesT()
    output_properties.colorSpace = metadata_schema.ColorSpaceType.GRAYSCALE
    output_content = metadata_schema.ContentT()
    output_content.contentPropertiesType = metadata_schema.ContentProperties.ImageProperties
    output_content.contentProperties = output_properties
    output_range = metadata_schema.ValueRangeT()
    output_range.min = 1
    output_range.max = 2
    output_content.range = output_range

    label_file = metadata_schema.AssociatedFileT()
    label_file.name = label_name
    label_file.description = "Labels for the background and foot confidence-mask channels."
    label_file.type = metadata_schema.AssociatedFileType.TENSOR_AXIS_LABELS

    output_metadata = metadata_schema.TensorMetadataT()
    output_metadata.name = "segmentation_masks"
    output_metadata.description = "Per-pixel softmax confidence for background and foot."
    output_metadata.stats = output_stats
    output_metadata.content = output_content
    output_metadata.associatedFiles = [label_file]

    subgraph_metadata = metadata_schema.SubGraphMetadataT()
    subgraph_metadata.inputTensorMetadata = [input_metadata]
    subgraph_metadata.outputTensorMetadata = [output_metadata]

    model_metadata = metadata_schema.ModelMetadataT()
    model_metadata.name = "FootSegmenter"
    model_metadata.description = (
        "MobileNetV2 U-Net binary foot segmenter trained on FootNet and specialized "
        "for dorsal smartphone views."
    )
    model_metadata.version = "1"
    model_metadata.author = "Foot Measure project; FootNet dataset creators credited in model card"
    model_metadata.license = "FootNet source data: CC BY 4.0"
    model_metadata.minParserVersion = "1.0.0"
    model_metadata.subgraphMetadata = [subgraph_metadata]

    builder = flatbuffers.Builder(0)
    builder.Finish(model_metadata.Pack(builder), METADATA_IDENTIFIER)
    return bytes(builder.Output())


def attach_metadata(model_bytes: bytes, label_bytes: bytes) -> tuple[bytes, dict[str, object]]:
    flatbuffers, metadata_schema, model_schema = _dependencies()
    if not model_schema.Model.ModelBufferHasIdentifier(model_bytes, 0):
        raise ValueError("Input is not a valid TFLite model")

    parsed = model_schema.Model.GetRootAsModel(model_bytes, 0)
    if parsed.SubgraphsLength() != 1:
        raise ValueError("MediaPipe Image Segmenter requires exactly one subgraph")
    subgraph = parsed.Subgraphs(0)
    if subgraph.InputsLength() != 1 or subgraph.OutputsLength() != 1:
        raise ValueError("MediaPipe Image Segmenter requires one input and one output")

    label_name = "labels.txt"
    metadata_bytes = _metadata_buffer(flatbuffers, metadata_schema, label_name)
    model = model_schema.ModelT.InitFromObj(parsed)
    metadata_storage = model_schema.BufferT()
    metadata_storage.data = bytearray(metadata_bytes)

    if model.buffers is None:
        model.buffers = []
    if model.metadata is None:
        model.metadata = []

    existing = next(
        (
            item
            for item in model.metadata
            if item.name is not None
            and (
                item.name.decode("utf-8")
                if isinstance(item.name, bytes)
                else str(item.name)
            )
            == METADATA_FIELD_NAME
        ),
        None,
    )
    if existing is None:
        model.buffers.append(metadata_storage)
        metadata_entry = model_schema.MetadataT()
        metadata_entry.name = METADATA_FIELD_NAME
        metadata_entry.buffer = len(model.buffers) - 1
        model.metadata.append(metadata_entry)
    else:
        model.buffers[existing.buffer] = metadata_storage

    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), TFLITE_IDENTIFIER)
    output = io.BytesIO(bytes(builder.Output()))
    # MediaPipe Tasks' metadata extractor expects associated files in an
    # uncompressed ZIP payload appended to the TFLite FlatBuffer.
    with zipfile.ZipFile(output, "a", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(label_name, label_bytes)
    populated = output.getvalue()

    reparsed = model_schema.Model.GetRootAsModel(populated, 0)
    metadata_index = None
    for index in range(reparsed.MetadataLength()):
        item = reparsed.Metadata(index)
        if item.Name().decode("utf-8") == METADATA_FIELD_NAME:
            metadata_index = item.Buffer()
            break
    if metadata_index is None:
        raise RuntimeError("Metadata entry was not written")
    stored_metadata = reparsed.Buffers(metadata_index).DataAsNumpy().tobytes()
    if not metadata_schema.ModelMetadata.ModelMetadataBufferHasIdentifier(
        stored_metadata, 0
    ):
        raise RuntimeError("Stored metadata does not have the M001 identifier")

    metadata_root = metadata_schema.ModelMetadata.GetRootAsModelMetadata(
        stored_metadata, 0
    )
    if metadata_root.SubgraphMetadataLength() != 1:
        raise RuntimeError("Stored metadata must describe one subgraph")
    stored_subgraph = metadata_root.SubgraphMetadata(0)
    if (
        stored_subgraph.InputTensorMetadataLength() != 1
        or stored_subgraph.OutputTensorMetadataLength() != 1
    ):
        raise RuntimeError("Stored metadata tensor counts do not match the model")
    stored_input = stored_subgraph.InputTensorMetadata(0)
    if stored_input.ProcessUnitsLength() != 1:
        raise RuntimeError("Input normalization metadata is missing")
    option_table = stored_input.ProcessUnits(0).Options()
    if option_table is None:
        raise RuntimeError("Input normalization options are missing")
    options = metadata_schema.NormalizationOptions()
    options.Init(option_table.Bytes, option_table.Pos)
    if options.MeanLength() != 1 or options.StdLength() != 1:
        raise RuntimeError("Input normalization metadata has the wrong shape")
    if abs(options.Mean(0) - 127.5) > 1e-6 or abs(options.Std(0) - 127.5) > 1e-6:
        raise RuntimeError("Input normalization metadata has the wrong values")

    with zipfile.ZipFile(io.BytesIO(populated)) as archive:
        if archive.read(label_name) != label_bytes:
            raise RuntimeError("Associated labels were not packed correctly")

    report = {
        "sha256": hashlib.sha256(populated).hexdigest(),
        "bytes": len(populated),
        "metadata_field": METADATA_FIELD_NAME,
        "metadata_identifier": METADATA_IDENTIFIER.decode("ascii"),
        "input_normalization": {"mean": [127.5], "std": [127.5]},
        "associated_files": [label_name],
        "labels": label_bytes.decode("utf-8").splitlines(),
    }
    return populated, report


def main() -> int:
    args = parse_args()
    model_bytes = args.model.read_bytes()
    label_bytes = args.labels.read_bytes()
    populated, report = attach_metadata(model_bytes, label_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(populated)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
