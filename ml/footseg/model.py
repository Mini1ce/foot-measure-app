"""MobileNetV2 U-Net model designed for built-in-only TFLite export."""

from __future__ import annotations

from numbers import Integral


def validate_image_size(image_size: int) -> int:
    """Return a valid square input size for the five-stage encoder/decoder."""
    if isinstance(image_size, bool) or not isinstance(image_size, Integral):
        raise ValueError("image_size must be an integer")
    image_size = int(image_size)
    if image_size <= 0 or image_size % 32 != 0:
        raise ValueError("image_size must be a positive multiple of 32")
    return image_size


def infer_image_size(input_shape: object) -> int:
    """Infer and validate the square RGB size from a Keras input shape."""
    if hasattr(input_shape, "as_list"):
        input_shape = input_shape.as_list()
    if (
        isinstance(input_shape, list)
        and input_shape
        and isinstance(input_shape[0], (list, tuple))
    ):
        if len(input_shape) != 1:
            raise ValueError("the segmentation model must have exactly one input")
        input_shape = input_shape[0]
    try:
        dimensions = tuple(input_shape)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"invalid model input shape: {input_shape!r}") from error
    if len(dimensions) != 4:
        raise ValueError(f"expected an NHWC model input, got {dimensions!r}")

    try:
        height = int(dimensions[1])
        width = int(dimensions[2])
        channels = int(dimensions[3])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"model height, width, and channels must be fixed, got {dimensions!r}"
        ) from error
    if height != width:
        raise ValueError(f"model input must be square, got {dimensions!r}")
    if channels != 3:
        raise ValueError(f"model input must have three RGB channels, got {dimensions!r}")
    return validate_image_size(height)


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required. Install ml/requirements-train.txt in Python 3.11."
        ) from error
    return tf


def build_foot_segmenter(
    *,
    image_size: int = 384,
    encoder_weights: str | None = "imagenet",
    freeze_encoder: bool = False,
):
    image_size = validate_image_size(image_size)
    tf = _require_tensorflow()
    inputs = tf.keras.Input((image_size, image_size, 3), name="image")
    encoder = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights=encoder_weights,
    )
    skip_names = [
        "block_1_expand_relu",
        "block_3_expand_relu",
        "block_6_expand_relu",
        "block_13_expand_relu",
    ]
    skips = [encoder.get_layer(name).output for name in skip_names]
    bridge = encoder.get_layer("block_16_project").output
    encoder.trainable = not freeze_encoder

    def decoder_block(value, skip, filters: int, name: str):
        value = tf.keras.layers.UpSampling2D(
            size=2, interpolation="bilinear", name=f"{name}_upsample"
        )(value)
        if skip is not None:
            value = tf.keras.layers.Concatenate(name=f"{name}_concat")([value, skip])
        value = tf.keras.layers.SeparableConv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv1"
        )(value)
        value = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(value)
        value = tf.keras.layers.ReLU(name=f"{name}_relu1")(value)
        value = tf.keras.layers.SeparableConv2D(
            filters, 3, padding="same", use_bias=False, name=f"{name}_conv2"
        )(value)
        value = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(value)
        return tf.keras.layers.ReLU(name=f"{name}_relu2")(value)

    value = bridge
    for index, (skip, filters) in enumerate(zip(reversed(skips), [256, 192, 128, 64]), start=1):
        value = decoder_block(value, skip, filters, f"decoder_{index}")
    value = decoder_block(value, None, 32, "decoder_5")
    logits = tf.keras.layers.Conv2D(2, 1, padding="same", name="logits")(value)
    outputs = tf.keras.layers.Softmax(axis=-1, dtype="float32", name="segmentation")(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="foot_segmenter_mobilenetv2_unet")
