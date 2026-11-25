"""Configuration and execution helpers for Ultralytics YOLO training."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import yaml


LOGGER = logging.getLogger(__name__)
DEFAULT_TRAIN_CONFIG = Path("configs/train.yaml")


class TrainingConfigurationError(ValueError):
    """Raised when training or dataset configuration is invalid."""


class TrainingError(RuntimeError):
    """Raised when the Ultralytics training process fails."""


@dataclass(frozen=True)
class TrainingConfig:
    """Validated options passed to Ultralytics."""

    model_weights: str
    epochs: int
    image_size: int
    batch_size: int
    device: str | int
    output_dir: Path
    run_name: str
    data_config_path: Path
    exist_ok: bool


@dataclass(frozen=True)
class DatasetConfig:
    """Validated dataset paths and class names for YOLO detection training."""

    train: str | list[str]
    val: str | list[str]
    names: list[str] | dict[int, str]
    test: str | list[str] | None = None

    def as_ultralytics_data(self) -> dict[str, Any]:
        """Return a dataset mapping with validated absolute image paths."""
        data: dict[str, Any] = {
            "train": self.train,
            "val": self.val,
            "names": self.names,
        }
        if self.test is not None:
            data["test"] = self.test
        return data


@dataclass(frozen=True)
class TrainingResult:
    """The trained Ultralytics model, its return value, and output location."""

    model: Any
    results: Any
    output_dir: Path


def _read_yaml_mapping(path: Path, description: str) -> dict[str, Any]:
    """Load a YAML mapping, providing contextual configuration errors."""
    try:
        with path.open(encoding="utf-8") as file:
            content = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise TrainingConfigurationError(
            f"{description} file does not exist: {path}"
        ) from exc
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrainingConfigurationError(
            f"Cannot read {description} file {path}: {exc}"
        ) from exc
    if not isinstance(content, dict):
        raise TrainingConfigurationError(
            f"{description} file must contain a YAML mapping: {path}"
        )
    return content


def load_training_config(path: str | Path = DEFAULT_TRAIN_CONFIG) -> TrainingConfig:
    """Load and validate training options from a YAML configuration file."""
    config_path = Path(path)
    raw = _read_yaml_mapping(config_path, "training configuration")
    allowed_keys = {
        "model_weights",
        "epochs",
        "image_size",
        "batch_size",
        "device",
        "output_dir",
        "run_name",
        "data_config",
        "exist_ok",
    }
    unknown_keys = set(raw) - allowed_keys
    if unknown_keys:
        raise TrainingConfigurationError(
            "Unknown training configuration key(s): "
            + ", ".join(sorted(str(key) for key in unknown_keys))
        )

    model_weights = raw.get("model_weights")
    if not isinstance(model_weights, str) or not model_weights.strip():
        raise TrainingConfigurationError(
            "model_weights must be a non-empty model name or path"
        )

    numeric_options: dict[str, int] = {}
    for key in ("epochs", "image_size", "batch_size"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TrainingConfigurationError(f"{key} must be a positive integer")
        numeric_options[key] = value

    device = raw.get("device")
    if isinstance(device, bool) or not isinstance(device, (str, int)):
        raise TrainingConfigurationError(
            "device must be 'cpu', a device index, or a device string"
        )
    if isinstance(device, int) and device < 0:
        raise TrainingConfigurationError("device index must be non-negative")
    if isinstance(device, str):
        device = device.strip()
        if not device:
            raise TrainingConfigurationError("device must not be empty")
        if device.lower() not in {"cpu", "mps"} and not re.fullmatch(
            r"\d+(,\d+)*",
            device,
        ):
            raise TrainingConfigurationError(
                "device must be 'cpu', 'mps', or a comma-separated list of "
                "non-negative device indices"
            )

    output_dir_value = raw.get("output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value.strip():
        raise TrainingConfigurationError("output_dir must be a non-empty path")
    project_root = config_path.resolve().parent.parent
    output_dir = Path(output_dir_value).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    if output_dir.exists() and not output_dir.is_dir():
        raise TrainingConfigurationError(
            f"output_dir exists but is not a directory: {output_dir}"
        )

    run_name = raw.get("run_name")
    if (
        not isinstance(run_name, str)
        or not run_name.strip()
        or Path(run_name).name != run_name
        or run_name in {".", ".."}
    ):
        raise TrainingConfigurationError(
            "run_name must be a non-empty directory name without path components"
        )

    data_config_value = raw.get("data_config")
    if not isinstance(data_config_value, str) or not data_config_value.strip():
        raise TrainingConfigurationError("data_config must be a non-empty path")
    data_config_path = Path(data_config_value).expanduser()
    if not data_config_path.is_absolute():
        data_config_path = config_path.resolve().parent / data_config_path

    exist_ok = raw.get("exist_ok", False)
    if not isinstance(exist_ok, bool):
        raise TrainingConfigurationError("exist_ok must be a boolean")

    return TrainingConfig(
        model_weights=model_weights.strip(),
        epochs=numeric_options["epochs"],
        image_size=numeric_options["image_size"],
        batch_size=numeric_options["batch_size"],
        device=device,
        output_dir=output_dir,
        run_name=run_name.strip(),
        data_config_path=data_config_path,
        exist_ok=exist_ok,
    )


def _resolve_dataset_path(value: Any, root: Path, key: str) -> str | list[str]:
    """Resolve and verify one YOLO image-directory value."""
    if isinstance(value, str) and value.strip():
        raw_paths = [value.strip()]
        is_list = False
    elif (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item.strip() for item in value)
    ):
        raw_paths = value
        is_list = True
    else:
        raise TrainingConfigurationError(
            f"dataset {key!r} must be a path or a non-empty list of paths"
        )

    resolved: list[str] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_dir():
            raise TrainingConfigurationError(
                f"dataset {key!r} path does not exist or is not a directory: {path}"
            )
        resolved.append(str(path))
    return resolved if is_list else resolved[0]


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load YOLO dataset configuration and verify all configured image paths.

    Relative dataset paths are resolved against the optional ``path`` root
    (itself relative to the YAML file's directory), or the YAML directory when
    no root is provided. Training and validation paths are required; test is
    optional.
    """
    config_path = Path(path).resolve()
    raw = _read_yaml_mapping(config_path, "dataset configuration")

    root_value = raw.get("path", ".")
    if not isinstance(root_value, str) or not root_value.strip():
        raise TrainingConfigurationError("dataset 'path' must be a non-empty path")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()

    train = _resolve_dataset_path(raw.get("train"), root, "train")
    val = _resolve_dataset_path(raw.get("val"), root, "val")
    test = (
        _resolve_dataset_path(raw["test"], root, "test")
        if "test" in raw and raw["test"] is not None
        else None
    )

    names_value = raw.get("names")
    if isinstance(names_value, list) and names_value and all(
        isinstance(name, str) and name.strip() for name in names_value
    ):
        names: list[str] | dict[int, str] = names_value
        class_count = len(names_value)
    elif isinstance(names_value, dict) and names_value:
        names = {}
        for key, name in names_value.items():
            try:
                if isinstance(key, bool) or not isinstance(key, (int, str)):
                    raise ValueError
                if isinstance(key, str) and not key.isdecimal():
                    raise ValueError
                class_id = int(key)
            except (TypeError, ValueError) as exc:
                raise TrainingConfigurationError(
                    "dataset class-name keys must be contiguous integers starting at 0"
                ) from exc
            if (
                class_id < 0
                or not isinstance(name, str)
                or not name.strip()
            ):
                raise TrainingConfigurationError(
                    "dataset class names must map non-negative integer IDs to non-empty strings"
                )
            names[class_id] = name
        if set(names) != set(range(len(names))):
            raise TrainingConfigurationError(
                "dataset class-name keys must be contiguous integers starting at 0"
            )
        class_count = len(names)
    else:
        raise TrainingConfigurationError(
            "dataset 'names' must be a non-empty list or integer-keyed mapping"
        )

    nc_value = raw.get("nc")
    if nc_value is not None and (
        isinstance(nc_value, bool)
        or not isinstance(nc_value, int)
        or nc_value != class_count
    ):
        raise TrainingConfigurationError(
            f"dataset 'nc' must equal the number of class names ({class_count})"
        )

    return DatasetConfig(train=train, val=val, test=test, names=names)


def _load_yolo_class():
    """Import Ultralytics lazily so configuration checks do not require it."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO training requires Ultralytics; install project dependencies "
            "with `python -m pip install -r requirements.txt`."
        ) from exc
    return YOLO


def _validate_output_location(
    output_dir: Path,
    run_name: str,
    dataset: DatasetConfig,
) -> Path:
    """Prevent Ultralytics output from being written inside a dataset split."""
    run_output_dir = (output_dir / run_name).resolve()
    dataset_paths = [dataset.train, dataset.val]
    if dataset.test is not None:
        dataset_paths.append(dataset.test)

    for configured_path in dataset_paths:
        paths = configured_path if isinstance(configured_path, list) else [configured_path]
        for value in paths:
            dataset_path = Path(value).resolve()
            if (
                run_output_dir == dataset_path
                or run_output_dir.is_relative_to(dataset_path)
                or dataset_path.is_relative_to(run_output_dir)
            ):
                raise TrainingConfigurationError(
                    "training output directory must not overlap a dataset split: "
                    f"{run_output_dir} and {dataset_path}"
                )
    return run_output_dir


def train_model(
    config_path: str | Path = DEFAULT_TRAIN_CONFIG,
    dataset_config_path: str | Path | None = None,
) -> TrainingResult:
    """Validate configuration and launch a YOLO training run.

    The dataset config defaults to ``data_config`` in the training YAML.
    Training results and checkpoints are written beneath the configured output
    directory by Ultralytics. Source dataset files are passed as read-only
    inputs and are not moved or modified by this module.

    Raises:
        TrainingConfigurationError: If configuration or dataset paths are invalid.
        ImportError: If Ultralytics is not installed.
        TrainingError: If model initialization or training fails.
    """
    config = load_training_config(config_path)
    selected_dataset_config = (
        Path(dataset_config_path).expanduser()
        if dataset_config_path is not None
        else config.data_config_path
    )
    if not selected_dataset_config.is_absolute():
        selected_dataset_config = Path.cwd() / selected_dataset_config
    dataset = load_dataset_config(selected_dataset_config)
    run_output_dir = _validate_output_location(
        config.output_dir,
        config.run_name,
        dataset,
    )

    LOGGER.info(
        "Starting YOLO training with model=%s, epochs=%d, image_size=%d, "
        "batch_size=%d, device=%s, dataset_config=%s, output_dir=%s/%s",
        config.model_weights,
        config.epochs,
        config.image_size,
        config.batch_size,
        config.device,
        selected_dataset_config,
        config.output_dir,
        config.run_name,
    )
    LOGGER.info(
        "Validated YOLO dataset paths: train=%s, val=%s, test=%s, classes=%s",
        dataset.train,
        dataset.val,
        dataset.test,
        dataset.names,
    )
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrainingError(
            f"Cannot create training output directory {config.output_dir}: {exc}"
        ) from exc

    try:
        yolo_class = _load_yolo_class()
        model = yolo_class(config.model_weights)
        results = model.train(
            data=dataset.as_ultralytics_data(),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            device=config.device,
            project=str(config.output_dir),
            name=config.run_name,
            exist_ok=config.exist_ok,
        )
    except ImportError:
        raise
    except Exception as exc:
        LOGGER.exception("YOLO training failed")
        raise TrainingError(f"YOLO training failed: {exc}") from exc

    LOGGER.info("YOLO training completed; outputs are in %s", run_output_dir)
    return TrainingResult(model, results, run_output_dir)
