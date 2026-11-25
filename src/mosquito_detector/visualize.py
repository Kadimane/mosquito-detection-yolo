"""Visualization helpers for YOLO detections and evaluation metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from PIL import ImageDraw, ImageFont

from .data.preprocess import read_image


_METRIC_NAMES = ("precision", "recall", "map50", "map50_95")


def _validate_output_path(source: Path, output_path: str | Path) -> Path:
    """Resolve a destination and ensure it cannot overwrite its source."""
    destination = Path(output_path).expanduser().resolve()
    if destination == source.expanduser().resolve():
        raise ValueError(f"output path must not overwrite source image: {source}")
    return destination


def save_annotated_image(
    source: str | Path,
    detections: tuple[object, ...] | list[object],
    output_path: str | Path,
) -> Path:
    """Draw detection boxes and labels on a copy of the source image.

    Detection entries must expose ``bbox`` (x1, y1, x2, y2), ``class_name``,
    and ``confidence`` attributes. The source image is never modified.
    """
    source_path = Path(source).expanduser().resolve()
    destination = _validate_output_path(source_path, output_path)
    image = read_image(source_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        coordinates = (x1, y1, x2, y2)
        if any(not math.isfinite(float(value)) for value in coordinates):
            raise ValueError("detection bounding-box coordinates must be finite")
        x1 = max(0, min(image.width - 1, round(float(x1))))
        y1 = max(0, min(image.height - 1, round(float(y1))))
        x2 = max(0, min(image.width - 1, round(float(x2))))
        y2 = max(0, min(image.height - 1, round(float(y2))))
        if x2 < x1 or y2 < y1:
            raise ValueError("detection bounding-box corners are invalid")
        confidence = float(detection.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("detection confidence must be finite and in [0, 1]")
        label = f"{detection.class_name} {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), label, fill="red", font=font)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def save_evaluation_plot(
    metrics: Mapping[str, float],
    output_path: str | Path,
) -> Path:
    """Save a bar chart for measured precision, recall and mAP metrics."""
    missing = set(_METRIC_NAMES) - set(metrics)
    if missing:
        raise ValueError(
            "evaluation metrics are missing: " + ", ".join(sorted(missing))
        )
    values = [float(metrics[name]) for name in _METRIC_NAMES]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("evaluation metrics must be finite values in [0, 1]")

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        raise ValueError("evaluation plot output must use PNG, JPEG, SVG, or PDF")
    destination.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    labels = ("Precision", "Recall", "mAP50", "mAP50-95")
    axis.bar(labels, values, color="#2878b5")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("YOLO evaluation metrics")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    try:
        figure.savefig(destination)
    finally:
        plt.close(figure)
    return destination
