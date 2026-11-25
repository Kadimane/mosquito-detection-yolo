"""Unit tests for YOLO training configuration and invocation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from src.mosquito_detector.train import (
    TrainingConfigurationError,
    TrainingError,
    load_dataset_config,
    load_training_config,
    train_model,
)


class TrainModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        self.dataset_root = self.root / "dataset"
        for split in ("train", "val", "test"):
            (self.dataset_root / split).mkdir(parents=True)

        self.train_config_path = self.config_dir / "train.yaml"
        self.dataset_config_path = self.config_dir / "data.yaml"
        self.write_training_config()
        self.write_dataset_config()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_training_config(self, **overrides) -> None:
        config = {
            "model_weights": "local-model.yaml",
            "epochs": 4,
            "image_size": 320,
            "batch_size": 2,
            "device": "cpu",
            "output_dir": "runs",
            "run_name": "unit-test",
            "data_config": "data.yaml",
            "exist_ok": False,
        }
        config.update(overrides)
        self.train_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def write_dataset_config(self, **overrides) -> None:
        config = {
            "path": "../dataset",
            "train": "train",
            "val": "val",
            "test": "test",
            "names": ["mosquito"],
        }
        config.update(overrides)
        self.dataset_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def test_load_training_config_resolves_paths_and_options(self) -> None:
        config = load_training_config(self.train_config_path)

        self.assertEqual(config.model_weights, "local-model.yaml")
        self.assertEqual(config.epochs, 4)
        self.assertEqual(config.image_size, 320)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.output_dir, self.root / "runs")
        self.assertEqual(config.data_config_path, self.dataset_config_path)
        self.assertFalse(config.exist_ok)

    def test_load_dataset_config_resolves_and_checks_all_split_paths(self) -> None:
        config = load_dataset_config(self.dataset_config_path)

        self.assertEqual(config.train, str(self.dataset_root / "train"))
        self.assertEqual(config.val, str(self.dataset_root / "val"))
        self.assertEqual(config.test, str(self.dataset_root / "test"))
        self.assertEqual(config.names, ["mosquito"])
        self.assertEqual(
            config.as_ultralytics_data(),
            {
                "train": str(self.dataset_root / "train"),
                "val": str(self.dataset_root / "val"),
                "test": str(self.dataset_root / "test"),
                "names": ["mosquito"],
            },
        )

    def test_train_uses_cpu_and_configured_ultralytics_arguments(self) -> None:
        yolo_model = MagicMock()
        yolo_model.train.return_value = {"metrics": "mocked"}
        yolo_class = MagicMock(return_value=yolo_model)

        with patch(
            "src.mosquito_detector.train._load_yolo_class",
            return_value=yolo_class,
        ):
            result = train_model(self.train_config_path)

        yolo_class.assert_called_once_with("local-model.yaml")
        call = yolo_model.train.call_args.kwargs
        self.assertEqual(call["epochs"], 4)
        self.assertEqual(call["imgsz"], 320)
        self.assertEqual(call["batch"], 2)
        self.assertEqual(call["device"], "cpu")
        self.assertEqual(call["project"], str(self.root / "runs"))
        self.assertEqual(call["name"], "unit-test")
        self.assertEqual(call["data"]["train"], str(self.dataset_root / "train"))
        self.assertEqual(result.results, {"metrics": "mocked"})
        self.assertEqual(result.output_dir, self.root / "runs" / "unit-test")

    def test_invalid_training_values_are_rejected(self) -> None:
        for key, value in (
            ("model_weights", ""),
            ("epochs", 0),
            ("epochs", True),
            ("image_size", -1),
            ("batch_size", 0),
            ("device", ""),
            ("device", -1),
            ("device", "gpu"),
            ("output_dir", ""),
            ("run_name", "../escape"),
            ("data_config", ""),
            ("exist_ok", "false"),
            ("unexpected", "value"),
        ):
            with self.subTest(key=key, value=value):
                self.write_training_config(**{key: value})
                with self.assertRaises(TrainingConfigurationError):
                    load_training_config(self.train_config_path)

    def test_missing_or_malformed_training_config_is_reported(self) -> None:
        with self.assertRaisesRegex(TrainingConfigurationError, "does not exist"):
            load_training_config(self.config_dir / "missing.yaml")

        self.train_config_path.write_text("epochs: [", encoding="utf-8")
        with self.assertRaisesRegex(TrainingConfigurationError, "Cannot read"):
            load_training_config(self.train_config_path)

    def test_missing_dataset_directory_fails_before_model_initialization(self) -> None:
        self.write_dataset_config(train="not-present")
        yolo_loader = MagicMock()

        with patch("src.mosquito_detector.train._load_yolo_class", yolo_loader):
            with self.assertRaisesRegex(TrainingConfigurationError, "train.*does not exist"):
                train_model(self.train_config_path)

        yolo_loader.assert_not_called()

    def test_invalid_dataset_names_and_missing_required_paths_are_rejected(self) -> None:
        for update in (
            {"train": None},
            {"train": ""},
            {"names": []},
            {"names": {0: "mosquito"}, "nc": 2},
            {"names": {1: "mosquito"}},
            {"names": {0: ""}},
            {"names": {0.5: "mosquito"}},
        ):
            with self.subTest(update=update):
                self.write_dataset_config(**update)
                with self.assertRaises(TrainingConfigurationError):
                    load_dataset_config(self.dataset_config_path)

    def test_training_errors_are_logged_and_wrapped(self) -> None:
        broken_yolo = MagicMock()
        broken_yolo.train.side_effect = RuntimeError("mock training failure")
        yolo_class = MagicMock(return_value=broken_yolo)

        with patch(
            "src.mosquito_detector.train._load_yolo_class",
            return_value=yolo_class,
        ):
            with self.assertLogs(
                "src.mosquito_detector.train",
                level="ERROR",
            ) as captured:
                with self.assertRaisesRegex(TrainingError, "mock training failure"):
                    train_model(self.train_config_path)
        self.assertIn("YOLO training failed", captured.output[0])

    def test_preexisting_output_file_is_rejected(self) -> None:
        output_file = self.root / "not-a-directory"
        output_file.write_text("file", encoding="utf-8")
        self.write_training_config(output_dir=str(output_file))

        with self.assertRaisesRegex(TrainingConfigurationError, "not a directory"):
            load_training_config(self.train_config_path)

    def test_output_must_not_overlap_dataset_directories(self) -> None:
        self.write_training_config(
            output_dir=str(self.dataset_root / "train"),
            run_name="training-output",
        )
        yolo_loader = MagicMock()

        with patch("src.mosquito_detector.train._load_yolo_class", yolo_loader):
            with self.assertRaisesRegex(TrainingConfigurationError, "must not overlap"):
                train_model(self.train_config_path)

        yolo_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
