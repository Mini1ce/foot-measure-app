"""Train the two-class foot segmentation model from a normalized manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from footseg.config import DEFAULT_CONFIG
from footseg.data import load_manifest_records, make_dataset
from footseg.losses import FootSegmentationLoss, custom_objects
from footseg.metrics import FootDice, FootIoU
from footseg.model import build_foot_segmenter, infer_image_size, validate_image_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/run"))
    parser.add_argument("--image-size", type=int, default=DEFAULT_CONFIG.image_size)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument(
        "--split-field",
        choices=("split", "pretrain_split"),
        default="split",
        help="Manifest field used for train/val assignment",
    )
    parser.add_argument("--views", nargs="*", help="Optional manifest views, e.g. dorsal")
    parser.add_argument("--sources", nargs="*", help="Optional source IDs")
    parser.add_argument("--subsets", nargs="*", help="Optional manifest subset tags")
    parser.add_argument(
        "--train-max-samples",
        type=int,
        help="Deterministically cap matching training samples",
    )
    parser.add_argument(
        "--validation-max-samples",
        type=int,
        help="Deterministically cap matching validation samples",
    )
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--geometry-augmentation",
        action="store_true",
        help="Enable paired rotation, scale, and translation augmentation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        args.image_size = validate_image_size(args.image_size)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        import tensorflow as tf
    except ImportError as error:
        raise SystemExit(
            "TensorFlow is not installed. Use Python 3.11 and install ml/requirements-train.txt"
        ) from error

    tf.keras.utils.set_random_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    filters = {
        "views": set(args.views) if args.views else None,
        "sources": set(args.sources) if args.sources else None,
        "subsets": set(args.subsets) if args.subsets else None,
        "synthetic": False if args.real_only else None,
    }
    try:
        train_records = load_manifest_records(
            args.manifest,
            "train",
            max_samples=args.train_max_samples,
            sample_seed=args.seed,
            split_field=args.split_field,
            **filters,
        )
        validation_records = load_manifest_records(
            args.manifest,
            "val",
            max_samples=args.validation_max_samples,
            sample_seed=args.seed,
            split_field=args.split_field,
            **filters,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    train_dataset = make_dataset(
        train_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=True,
        seed=args.seed,
        geometry_augmentation=args.geometry_augmentation,
    )
    validation_dataset = make_dataset(
        validation_records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        training=False,
        seed=args.seed,
    )

    if args.initial_model:
        model = tf.keras.models.load_model(
            args.initial_model,
            custom_objects=custom_objects(),
            compile=False,
        )
        try:
            initial_image_size = infer_image_size(model.input_shape)
        except ValueError as error:
            raise SystemExit(f"Invalid initial model: {error}") from error
        if initial_image_size != args.image_size:
            raise SystemExit(
                "Initial model input size "
                f"{initial_image_size} does not match --image-size {args.image_size}"
            )
    else:
        model = build_foot_segmenter(
            image_size=args.image_size,
            encoder_weights=None if args.no_imagenet else "imagenet",
            freeze_encoder=args.freeze_encoder,
        )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=FootSegmentationLoss(),
        metrics=[FootDice(), FootIoU()],
    )
    checkpoint = args.output / "best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_foot_dice",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_foot_dice",
            mode="max",
            patience=DEFAULT_CONFIG.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_foot_dice",
            mode="max",
            factor=0.35,
            patience=4,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(args.output / "training.csv"),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        steps_per_epoch=math.ceil(len(train_records) / args.batch_size),
        validation_steps=math.ceil(len(validation_records) / args.batch_size),
        callbacks=callbacks,
    )
    model.save(args.output / "final.keras")
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "train_samples": len(train_records),
                "validation_samples": len(validation_records),
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "split_field": args.split_field,
                "train_max_samples": args.train_max_samples,
                "validation_max_samples": args.validation_max_samples,
                "geometry_augmentation": args.geometry_augmentation,
                "filters": {key: sorted(value) if isinstance(value, set) else value for key, value in filters.items()},
                "history": history.history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training complete. Best model: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
