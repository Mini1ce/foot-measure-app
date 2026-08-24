"""Serializable foreground Dice and IoU Keras metrics."""

from __future__ import annotations


try:
    import tensorflow as tf
except ImportError:
    tf = None


if tf is not None:

    class _BinaryFootMetric(tf.keras.metrics.Metric):
        def __init__(self, metric_name: str, name: str, **kwargs):
            super().__init__(name=name, **kwargs)
            self.metric_name = metric_name
            self.intersection = self.add_weight(name="intersection", initializer="zeros")
            self.predicted = self.add_weight(name="predicted", initializer="zeros")
            self.actual = self.add_weight(name="actual", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            truth = tf.cast(tf.equal(tf.cast(y_true, tf.int32), 1), tf.float32)
            prediction = tf.cast(tf.argmax(y_pred, axis=-1) == 1, tf.float32)
            if sample_weight is not None:
                weight = tf.cast(sample_weight, tf.float32)
                truth *= weight
                prediction *= weight
            self.intersection.assign_add(tf.reduce_sum(truth * prediction))
            self.predicted.assign_add(tf.reduce_sum(prediction))
            self.actual.assign_add(tf.reduce_sum(truth))

        def result(self):
            epsilon = tf.keras.backend.epsilon()
            if self.metric_name == "dice":
                return (2.0 * self.intersection + epsilon) / (
                    self.predicted + self.actual + epsilon
                )
            union = self.predicted + self.actual - self.intersection
            return (self.intersection + epsilon) / (union + epsilon)

        def reset_state(self):
            for value in (self.intersection, self.predicted, self.actual):
                value.assign(0.0)

        def get_config(self):
            # ``metric_name`` is fixed by each public subclass. Including it in
            # the serialized config would make Keras pass it back through
            # ``**kwargs`` while the subclass also supplies it positionally.
            return super().get_config()


    @tf.keras.utils.register_keras_serializable(package="FootMeasure")
    class FootDice(_BinaryFootMetric):
        def __init__(self, name: str = "foot_dice", **kwargs):
            super().__init__("dice", name=name, **kwargs)


    @tf.keras.utils.register_keras_serializable(package="FootMeasure")
    class FootIoU(_BinaryFootMetric):
        def __init__(self, name: str = "foot_iou", **kwargs):
            super().__init__("iou", name=name, **kwargs)

else:

    class FootDice:  # pragma: no cover
        pass

    class FootIoU:  # pragma: no cover
        pass
