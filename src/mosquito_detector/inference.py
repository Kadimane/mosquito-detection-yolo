"""Image inference using trained Ultralytics YOLO weights."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

from .data.preprocess import _IMAGE_EXTENSIONS, list_image_files, read_image
from .visualize import save_annotated_image


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A detected object with a pixel-space ``(x1, y1, x2, y2)`` box."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageInference:
    """Detections and optional annotated-image output for one source image."""

    source: Path
    width: int
    height: int
    detections: tuple[Detection, ...]
    annotated_image: Path | None


def _load_yolo_class():
    """Import Ultralytics only when inference is requested."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO inference requires Ultralytics; install project dependencies "
            "with `python -m pip install -r requirements.txt`."
        ) from exc
    return YOLO


def _to_list(values: Any) -> list[Any]:
    """Convert Ultralytics tensor-like values to Python lists."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def _class_name(names: Any, class_id: int) -> str:
    """Resolve a class name from Ultralytics' list/dictionary representation."""
    if isinstance(names, dict):
        value = names.get(class_id, str(class_id))
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        value = names[class_id]
    else:
        value = str(class_id)
    return str(value)


def _parse_detections(result: Any) -> tuple[Detection, ...]:
    """Convert one Ultralytics result object to typed detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise ValueError("YOLO prediction result does not contain boxes")
    if len(boxes) == 0:
        return ()

    coordinates = _to_list(boxes.xyxy)
    confidences = _to_list(boxes.conf)
    class_ids = _to_list(boxes.cls)
    if not (len(coordinates) == len(confidences) == len(class_ids)):
        raise ValueError("YOLO result has inconsistent box, confidence, and class counts")

    names = getattr(result, "names", {})
    detections: list[Detection] = []
    for raw_box, raw_confidence, raw_class_id in zip(
        coordinates,
        confidences,
        class_ids,
    ):
        if len(raw_box) != 4:
            raise ValueError("YOLO result contains a bounding box without four coordinates")
        bbox = tuple(float(value) for value in raw_box)
        confidence = float(raw_confidence)
        class_id_value = float(raw_class_id)
        if (
            any(not math.isfinite(value) for value in bbox)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not math.isfinite(class_id_value)
            or not class_id_value.is_integer()
            or class_id_value < 0
            or bbox[2] < bbox[0]
            or bbox[3] < bbox[1]
        ):
            raise ValueError("YOLO result contains invalid detection values")
        class_id = int(class_id_value)
        detections.append(
            Detection(
                class_id,
                _class_name(names, class_id),
                confidence,
                (bbox[0], bbox[1], bbox[2], bbox[3]),
            )
        )
    return tuple(detections)


def _validate_device(device: str | int) -> None:
    """Validate device values before calling Ultralytics."""
    if isinstance(device, bool) or not isinstance(device, (str, int)):
        raise ValueError("device must be 'cpu', a device index, or a device string")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("device index must be non-negative")
    elif not device.strip():
        raise ValueError("device must not be empty")


def run_inference(
    weights: str | Path,
    source: str | Path,
    output_dir: str | Path | None = "outputs/predictions",
    confidence: float = 0.25,
    image_size: int = 640,
    device: str | int = "cpu",
) -> tuple[ImageInference, ...]:
    """Run YOLO on one image or every supported image in a directory.

    Trained weights must be an existing file. Annotated copies are written to
    ``output_dir`` (or omitted by passing ``None``); source files are not changed.
    """
    weight_path = Path(weights).expanduser()
    if not weight_path.is_file():
        raise FileNotFoundError(f"YOLO weights file does not exist: {weight_path}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("confidence must be a finite number in [0, 1]")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    _validate_device(device)

    source_path = Path(source).expanduser()
    if source_path.is_dir():
        image_paths = list_image_files(source_path)
        if not image_paths:
            raise ValueError(f"No supported images found in source directory: {source_path}")
    elif source_path.is_file():
        if source_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {source_path.suffix}")
        image_paths = (source_path,)
    else:
        raise FileNotFoundError(f"Image source does not exist: {source_path}")

    image_sizes: dict[Path, tuple[int, int]] = {}
    for image_path in image_paths:
        image = read_image(image_path)
        image_sizes[image_path] = image.size

    destination: Path | None = None
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists() and not destination.is_dir():
            raise ValueError(
                f"inference output path is not a directory: {destination}"
            )
        for image_path in image_paths:
            resolved_source = image_path.resolve()
            if (
                destination == resolved_source.parent
                or destination == resolved_source
                or destination.is_relative_to(resolved_source.parent)
            ):
                raise ValueError(
                    f"output directory must be separate from source images: {destination}"
                )

    LOGGER.info(
        "Running YOLO inference for %d image(s), weights=%s, device=%s",
        len(image_paths),
        weight_path,
        device,
    )
    try:
        yolo_class = _load_yolo_class()
        model = yolo_class(str(weight_path))
        raw_results = model.predict(
            source=[str(path) for path in image_paths],
            conf=float(confidence),
            imgsz=image_size,
            device=device,
            verbose=False,
        )
    except ImportError:
        raise
    except Exception as exc:
        LOGGER.exception("YOLO inference failed")
        raise RuntimeError(f"YOLO inference failed: {exc}") from exc

    if raw_results is None or len(raw_results) != len(image_paths):
        raise RuntimeError(
            "YOLO inference returned a different number of results than input images"
        )

    results: list[ImageInference] = []
    for image_path, raw_result in zip(image_paths, raw_results):
        detections = _parse_detections(raw_result)
        annotated_path = None
        if destination is not None:
            annotated_path = save_annotated_image(
                image_path,
                detections,
                destination / image_path.name,
            )
        width, height = image_sizes[image_path]
        results.append(
            ImageInference(
                image_path,
                width,
                height,
                detections,
                annotated_path,
            )
        )
    return tuple(results)
