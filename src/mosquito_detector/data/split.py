"""Deterministic, non-destructive splitting for YOLO detection datasets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

from .preprocess import (
    list_annotation_files,
    list_image_files,
    match_image_and_annotation_files,
)


@dataclass(frozen=True)
class DatasetItem:
    """An image assigned to a split and its optional annotation file."""

    image_path: Path
    annotation_path: Path | None


@dataclass(frozen=True)
class DatasetSplit:
    """Image assignments and pairing issues from a dataset split operation."""

    train: tuple[DatasetItem, ...]
    validation: tuple[DatasetItem, ...]
    test: tuple[DatasetItem, ...]
    missing_annotations: tuple[Path, ...]
    unmatched_annotations: tuple[Path, ...]

    def summary(self) -> str:
        """Return image and annotation counts for each split."""
        return "\n".join(
            (
                f"Train: {len(self.train)} images "
                f"({sum(item.annotation_path is not None for item in self.train)} "
                "with annotations)",
                f"Validation: {len(self.validation)} images "
                f"({sum(item.annotation_path is not None for item in self.validation)} "
                "with annotations)",
                f"Test: {len(self.test)} images "
                f"({sum(item.annotation_path is not None for item in self.test)} "
                "with annotations)",
                f"Images missing annotations: {len(self.missing_annotations)}",
                f"Unmatched annotation files: {len(self.unmatched_annotations)}",
            )
        )


def _validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    """Validate and normalize the split ratio inputs."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        for ratio in ratios
    ):
        raise ValueError("split ratios must be finite numbers in [0, 1]")
    try:
        normalized = tuple(float(ratio) for ratio in ratios)
    except OverflowError as exc:
        raise ValueError("split ratios must be finite numbers in [0, 1]") from exc
    if any(not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in normalized):
        raise ValueError("split ratios must be finite numbers in [0, 1]")
    if not math.isclose(sum(normalized), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("train, validation, and test ratios must sum to 1")
    return normalized


def _split_counts(
    item_count: int,
    ratios: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Allocate counts by largest remainder, preserving the total item count."""
    exact_counts = tuple(item_count * ratio for ratio in ratios)
    counts = [math.floor(count) for count in exact_counts]
    remaining = item_count - sum(counts)
    remainders = sorted(
        range(len(ratios)),
        key=lambda index: (-(exact_counts[index] - counts[index]), index),
    )
    for index in remainders[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def split_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplit:
    """Partition dataset paths reproducibly without moving or copying files.

    Each supported image appears in exactly one returned split. Its annotation
    path is included when a case-insensitive filename-stem match exists;
    images without labels are still assigned and reported. Annotation files
    without matching images are reported separately because they cannot be
    assigned to an image split.

    The operation returns assignments only and never modifies source files.
    Split sizes use largest-remainder allocation, so their counts sum exactly
    to the number of source images.

    Args:
        images_dir: Directory containing source images.
        labels_dir: Directory containing YOLO ``.txt`` annotations.
        train_ratio: Training fraction. Defaults to 0.7.
        validation_ratio: Validation fraction. Defaults to 0.2.
        test_ratio: Test fraction. Defaults to 0.1.
        seed: Integer seed controlling deterministic shuffling.

    Raises:
        ValueError: If ratios are invalid, directories are unavailable, or
            case-insensitive duplicate stems make matching ambiguous.
    """
    ratios = _validate_ratios(train_ratio, validation_ratio, test_ratio)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    image_paths = list_image_files(images_dir)
    annotation_paths = list_annotation_files(labels_dir)
    matches = match_image_and_annotation_files(image_paths, annotation_paths)

    annotation_by_image = {
        image_path: annotation_path for image_path, annotation_path in matches.matches
    }
    items = [
        DatasetItem(image_path, annotation_by_image.get(image_path))
        for image_path in image_paths
    ]
    random.Random(seed).shuffle(items)

    train_count, validation_count, test_count = _split_counts(len(items), ratios)
    validation_end = train_count + validation_count
    test_end = validation_end + test_count
    return DatasetSplit(
        train=tuple(items[:train_count]),
        validation=tuple(items[train_count:validation_end]),
        test=tuple(items[validation_end:test_end]),
        missing_annotations=matches.missing_annotations,
        unmatched_annotations=matches.unmatched_annotations,
    )
