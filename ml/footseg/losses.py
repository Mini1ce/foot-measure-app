"""Serializable segmentation loss for sparse two-class masks."""

from __future__ import annotations


def custom_objects() -> dict[str, object]:
    from .metrics import FootDice, FootIoU

    return {
        "FootSegmentationLoss": FootSegmentationLoss,
        "FootDice": FootDice,
        "FootIoU": FootIoU,
    }


try:
    import tensorflow as tf
except ImportError:  # Allow lightweight dataset tools to import the package.
    tf = None


if tf is not None:

    @tf.keras.utils.register_keras_serializable(package="FootMeasure")
    class FootSegmentationLoss(tf.keras.losses.Loss):
        def __init__(
            self,
            cross_entropy_weight: float = 0.5,
            name: str = "foot_segmentation_loss",
            reduction: str = "sum_over_batch_size",
            **kwargs,
        ):
            super().__init__(name=name, reduction=reduction, **kwargs)
            self.cross_entropy_weight = cross_entropy_weight
            self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy()

        def call(self, y_true, y_pred):
            y_true = tf.cast(y_true, tf.int32)
            cross_entropy = self.cross_entropy(y_true, y_pred)
            foot_truth = tf.cast(tf.equal(y_true, 1), tf.float32)
            foot_prediction = tf.cast(y_pred[..., 1], tf.float32)
            axes = tuple(range(1, len(foot_prediction.shape)))
            intersection = tf.reduce_sum(foot_truth * foot_prediction, axis=axes)
            denominator = tf.reduce_sum(foot_truth + foot_prediction, axis=axes)
            dice_loss = 1.0 - tf.reduce_mean((2.0 * intersection + 1.0) / (denominator + 1.0))
            return self.cross_entropy_weight * cross_entropy + (1.0 - self.cross_entropy_weight) * dice_loss

        def get_config(self):
            return {**super().get_config(), "cross_entropy_weight": self.cross_entropy_weight}

else:

    class FootSegmentationLoss:  # pragma: no cover - only used to explain a missing dependency
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorFlow is required to construct FootSegmentationLoss")
