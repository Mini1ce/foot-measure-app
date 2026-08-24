"""Shared, dependency-light configuration for the segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    image_size: int = 384
    batch_size: int = 4
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    seed: int = 20260822
    num_classes: int = 2


DEFAULT_CONFIG = TrainingConfig()
