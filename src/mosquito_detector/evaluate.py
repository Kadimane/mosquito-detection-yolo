"""YOLO object-detection model evaluation and metric reporting."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any

from .train import DatasetConfig, TrainingConfigurationError, load_dataset_config
from .visualize import save_evaluation_plot


LOGGER = logging.getLogger(__name__)
_METRIC_ALIASES = {
    "precision": ("metrics/precision(B)", "precision"),
    "recall": ("metrics/recall(B)", "recall"),
    "map50": ("metrics/mAP50(B)", "map50", "mAP50"),
    "map50_95": ("metrics/mAP50-95(B)", "map50_95", "mAP50-95"),
}


@dataclass(frozen=True)
class EvaluationMetrics:
    """Measured detection metrics from an Ultralytics validation run."""

    precision: float
    recall: float
    map50: float
    map50_95: float

    def as_dict(self) -> dict[str, float]:
        """Return metrics with stable keys for display or serialization."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Measured metrics, the raw Ultralytics result, and saved plot path."""

    metrics: EvaluationMetrics
    raw_results: Any
    plot_path: Path
    split: str


def _load_yolo_class():
    """Import Ultralytics only when evaluation is requested."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO evaluation requires Ultralytics; install project dependencies "
            "with `python -m pip install -r requirements.txt`."
        ) from exc
    return YOLO


def _metric_value(results: Any, metric: str) -> float:
    """Extract one metric from the Ultralytics results dictionary or box stats."""
    result_mapping = getattr(results, "results_dict", None)
    value = math.nan
    if isinstance(result_mapping, dict):
        for key in _METRIC_ALIASES[metric]:
            if key in result_mapping:
                value = float(result_mapping[key])
                break
    if not math.isfinite(value):
        box_metrics = getattr(results, "box", None)
        attributes = {
            "precision": "mp",
            "recall": "mr",
            "map50": "map50",
            "map50_95": "map",
        }
        raw_value = getattr(box_metrics, attributes[metric], math.nan)
        value = float(raw_value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError(
            f"YOLO evaluation did not return a valid measured {metric} value"
        )
    return value


def _validate_evaluation_options(
    image_size: int,
    batch_size: int,
    device: str | int,
) -> None:
    """Reject invalid evaluator options before loading a model."""
    for name, value in (("image_size", image_size), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(device, bool) or not isinstance(device, (str, int)):
        raise ValueError("device must be 'cpu', a device index, or a device string")
    if isinstance(device, int) and device < 0:
        raise ValueError("device index must be non-negative")
    if isinstance(device, str) and not device.strip():
        raise ValueError("device must not be empty")


def evaluate_model(
    weights: str | Path,
    dataset_config_path: str | Path,
    output_dir: str | Path = "outputs/plots",
    split: str | None = None,
    image_size: int = 640,
    batch_size: int = 8,
    device: str | int = "cpu",
) -> EvaluationResult:
    """Evaluate trained weights and save a chart of measured metrics.

    Defaults to the test set when configured, otherwise the validation set.
    Dataset configuration and directories are validated before the model is
    loaded. No training or data mutation is performed.
    """
    _validate_evaluation_options(image_size, batch_size, device)
    weights_path = Path(weights).expanduser()
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights file does not exist: {weights_path}")
    config_path = Path(dataset_config_path).expanduser()
    dataset: DatasetConfig = load_dataset_config(config_path)
    selected_split = split or ("test" if dataset.test is not None else "val")
    if selected_split not in {"train", "val", "test"}:
        raise ValueError("split must be one of 'train', 'val', or 'test'")
    if selected_split == "test" and dataset.test is None:
        raise TrainingConfigurationError(
            "test split requested but no test path is configured in the dataset YAML"
        )

    selected_path = getattr(dataset, selected_split)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"evaluation output path is not a directory: {output_path}")
    source_paths = [dataset.train, dataset.val]
    if dataset.test is not None:
        source_paths.append(dataset.test)
    for source_value in source_paths:
        paths = source_value if isinstance(source_value, list) else [source_value]
        for value in paths:
            source_path = Path(value).resolve()
            if (
                output_path == source_path
                or output_path.is_relative_to(source_path)
                or source_path.is_relative_to(output_path)
            ):
                raise ValueError(
                    "evaluation output directory must not overlap dataset paths: "
                    f"{output_path} and {source_path}"
                )

    LOGGER.info(
        "Evaluating YOLO weights=%s on split=%s at %s using device=%s",
        weights_path,
        selected_split,
        selected_path,
        device,
    )
    run_name = f"evaluation-{selected_split}"
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        model = _load_yolo_class()(str(weights_path))
        raw_results = model.val(
            data=dataset.as_ultralytics_data(),
            split=selected_split,
            imgsz=image_size,
            batch=batch_size,
            device=device,
            plots=True,
            project=str(output_path),
            name=run_name,
            exist_ok=True,
        )
    except ImportError:
        raise
    except Exception as exc:
        LOGGER.exception("YOLO evaluation failed")
        raise RuntimeError(f"YOLO evaluation failed: {exc}") from exc

    metrics = EvaluationMetrics(
        precision=_metric_value(raw_results, "precision"),
        recall=_metric_value(raw_results, "recall"),
        map50=_metric_value(raw_results, "map50"),
        map50_95=_metric_value(raw_results, "map50_95"),
    )
    plot_path = save_evaluation_plot(
        metrics.as_dict(),
        output_path / run_name / "metrics.png",
    )
    LOGGER.info("YOLO evaluation metrics: %s", metrics.as_dict())
    return EvaluationResult(metrics, raw_results, plot_path, selected_split)
