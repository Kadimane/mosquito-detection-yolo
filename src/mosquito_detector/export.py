"""Export locally available Ultralytics YOLO weights to ONNX."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class ModelExportError(RuntimeError):
    """Raised when YOLO model export fails or produces no ONNX model."""


@dataclass(frozen=True)
class ExportResult:
    """Details of a successful ONNX export."""

    weights_path: Path
    onnx_path: Path
    image_size: int
    options: dict[str, Any]


def _load_yolo_class():
    """Import Ultralytics only when an export is requested."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires Ultralytics; install project dependencies "
            "with `python -m pip install -r requirements.txt`."
        ) from exc
    return YOLO


def export_to_onnx(
    weights: str | Path,
    output_path: str | Path | None = None,
    image_size: int = 640,
    *,
    dynamic: bool = False,
    simplify: bool = False,
    opset: int | None = None,
    half: bool = False,
    device: str | int = "cpu",
    overwrite: bool = False,
) -> ExportResult:
    """Export an existing local YOLO weights file as ONNX.

    Ultralytics may initially write the exported file beside the weights. If an
    explicit output location is given, the generated ONNX file is moved there;
    the source weights are never modified. The exporter is called lazily and
    is not imported until validation has passed.
    """
    weights_path = Path(weights).expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights file does not exist: {weights_path}")
    if weights_path.suffix.lower() == ".onnx":
        raise ValueError("weights must be a source model file, not an ONNX file")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    for name, value in (
        ("dynamic", dynamic),
        ("simplify", simplify),
        ("half", half),
        ("overwrite", overwrite),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    if opset is not None and (
        isinstance(opset, bool) or not isinstance(opset, int) or opset <= 0
    ):
        raise ValueError("opset must be a positive integer or None")
    if isinstance(device, bool) or not isinstance(device, (str, int)):
        raise ValueError("device must be 'cpu', a device index, or a device string")
    if isinstance(device, int) and device < 0:
        raise ValueError("device index must be non-negative")
    if isinstance(device, str) and not device.strip():
        raise ValueError("device must not be empty")

    target_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else weights_path.with_suffix(".onnx")
    )
    if target_path.suffix.lower() != ".onnx":
        raise ValueError("output_path must have an .onnx extension")
    if target_path == weights_path:
        raise ValueError("output_path must not overwrite the source weights")
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"ONNX output already exists (set overwrite=True to replace it): {target_path}"
        )
    if target_path.exists() and not target_path.is_file():
        raise ValueError(f"ONNX output path is not a file: {target_path}")
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelExportError(
            f"Cannot create ONNX output directory {target_path.parent}: {exc}"
        ) from exc

    options: dict[str, Any] = {
        "format": "onnx",
        "imgsz": image_size,
        "dynamic": dynamic,
        "simplify": simplify,
        "half": half,
        "device": device,
    }
    if opset is not None:
        options["opset"] = opset

    LOGGER.info(
        "Exporting YOLO weights=%s to ONNX at %s (image_size=%d, device=%s)",
        weights_path,
        target_path,
        image_size,
        device,
    )
    try:
        model = _load_yolo_class()(str(weights_path))
        exported_value = model.export(**options)
        if not isinstance(exported_value, (str, Path)):
            raise ModelExportError(
                "Ultralytics did not return the path to its ONNX export"
            )
        exported_path = Path(exported_value).expanduser().resolve()
        if exported_path.suffix.lower() != ".onnx" or not exported_path.is_file():
            raise ModelExportError(
                f"Ultralytics ONNX export was not found at {exported_path}"
            )
        if exported_path != target_path:
            if target_path.exists() and not overwrite:
                raise FileExistsError(
                    f"ONNX output already exists: {target_path}"
                )
            os.replace(exported_path, target_path)
    except ImportError:
        raise
    except Exception as exc:
        LOGGER.exception("YOLO ONNX export failed")
        if isinstance(exc, ModelExportError):
            raise
        raise ModelExportError(f"YOLO ONNX export failed: {exc}") from exc

    LOGGER.info("YOLO ONNX export saved to %s", target_path)
    return ExportResult(weights_path, target_path, image_size, options)
