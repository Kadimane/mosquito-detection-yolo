"""Mocked tests for inference, evaluation, and visualization workflows."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import yaml

from src.mosquito_detector.evaluate import evaluate_model
from src.mosquito_detector.inference import Detection, run_inference
from src.mosquito_detector.train import TrainingConfigurationError
from src.mosquito_detector.visualize import (
    save_annotated_image,
    save_evaluation_plot,
)


class InferenceEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"mock weights")
        self.images_dir = self.root / "images"
        self.images_dir.mkdir()
        self.image_path = self.images_dir / "mosquito.png"
        Image.new("RGB", (32, 24), (230, 230, 230)).save(self.image_path)

        self.dataset_root = self.root / "dataset"
        for split in ("train", "val", "test"):
            (self.dataset_root / split).mkdir(parents=True)
        self.dataset_config = self.root / "data.yaml"
        self.dataset_config.write_text(
            yaml.safe_dump(
                {
                    "path": str(self.dataset_root),
                    "train": "train",
                    "val": "val",
                    "test": "test",
                    "names": ["mosquito"],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def prediction_result():
        boxes = MagicMock()
        boxes.__len__.return_value = 1
        boxes.xyxy = [[2.0, 3.0, 20.0, 18.0]]
        boxes.conf = [0.875]
        boxes.cls = [0]
        result = MagicMock()
        result.boxes = boxes
        result.names = {0: "mosquito"}
        return result

    def test_inference_returns_named_detections_and_saves_annotated_copy(self) -> None:
        yolo_model = MagicMock()
        yolo_model.predict.return_value = [self.prediction_result()]
        yolo_class = MagicMock(return_value=yolo_model)
        output_dir = self.root / "predictions"
        original_bytes = self.image_path.read_bytes()

        with patch(
            "src.mosquito_detector.inference._load_yolo_class",
            return_value=yolo_class,
        ):
            results = run_inference(
                self.weights,
                self.image_path,
                output_dir,
                device="cpu",
            )

        yolo_class.assert_called_once_with(str(self.weights))
        kwargs = yolo_model.predict.call_args.kwargs
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["source"], [str(self.image_path)])
        self.assertEqual(results[0].width, 32)
        self.assertEqual(results[0].height, 24)
        self.assertEqual(
            results[0].detections,
            (Detection(0, "mosquito", 0.875, (2.0, 3.0, 20.0, 18.0)),),
        )
        self.assertTrue(results[0].annotated_image.is_file())
        self.assertEqual(self.image_path.read_bytes(), original_bytes)

    def test_inference_rejects_missing_weights_and_invalid_options_before_model_load(self) -> None:
        loader = MagicMock()
        with patch("src.mosquito_detector.inference._load_yolo_class", loader):
            with self.assertRaises(FileNotFoundError):
                run_inference(self.root / "absent.pt", self.image_path)
            for kwargs in (
                {"confidence": 1.1},
                {"confidence": float("nan")},
                {"image_size": 0},
                {"device": ""},
            ):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaises(ValueError):
                        run_inference(self.weights, self.image_path, **kwargs)
        loader.assert_not_called()

    def test_inference_rejects_empty_directories_and_output_inside_source(self) -> None:
        empty_directory = self.root / "empty"
        empty_directory.mkdir()
        with self.assertRaisesRegex(ValueError, "No supported images"):
            run_inference(self.weights, empty_directory, output_dir=None)

        with self.assertRaisesRegex(ValueError, "separate from source"):
            run_inference(
                self.weights,
                self.image_path,
                output_dir=self.images_dir,
            )

    def test_evaluation_reports_measured_metrics_and_saves_plot(self) -> None:
        yolo_model = MagicMock()
        yolo_model.val.return_value.results_dict = {
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.7,
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": 0.55,
        }
        yolo_class = MagicMock(return_value=yolo_model)
        output_dir = self.root / "evaluation"

        with patch(
            "src.mosquito_detector.evaluate._load_yolo_class",
            return_value=yolo_class,
        ):
            result = evaluate_model(
                self.weights,
                self.dataset_config,
                output_dir=output_dir,
            )

        self.assertEqual(result.split, "test")
        self.assertEqual(result.metrics.precision, 0.8)
        self.assertEqual(result.metrics.recall, 0.7)
        self.assertEqual(result.metrics.map50, 0.75)
        self.assertEqual(result.metrics.map50_95, 0.55)
        self.assertTrue(result.plot_path.is_file())
        self.assertEqual(yolo_model.val.call_args.kwargs["device"], "cpu")
        self.assertEqual(yolo_model.val.call_args.kwargs["split"], "test")
        self.assertEqual(
            yolo_model.val.call_args.kwargs["project"],
            str(output_dir.resolve()),
        )
        self.assertEqual(
            yolo_model.val.call_args.kwargs["name"],
            "evaluation-test",
        )

    def test_evaluation_uses_box_metric_properties_and_rejects_unavailable_test(self) -> None:
        yolo_model = MagicMock()
        result = MagicMock()
        result.results_dict = None
        result.box.mp = 0.8
        result.box.mr = 0.6
        result.box.map50 = 0.7
        result.box.map = 0.4
        yolo_model.val.return_value = result
        yolo_class = MagicMock(return_value=yolo_model)
        config_without_test = self.root / "without-test.yaml"
        config_without_test.write_text(
            yaml.safe_dump(
                {
                    "path": str(self.dataset_root),
                    "train": "train",
                    "val": "val",
                    "names": ["mosquito"],
                }
            ),
            encoding="utf-8",
        )

        with patch(
            "src.mosquito_detector.evaluate._load_yolo_class",
            return_value=yolo_class,
        ):
            with self.assertRaisesRegex(TrainingConfigurationError, "no test path"):
                evaluate_model(self.weights, config_without_test, split="test")
            result = evaluate_model(
                self.weights,
                config_without_test,
                output_dir=self.root / "validation",
            )

        self.assertEqual(result.split, "val")
        self.assertEqual(result.metrics.map50_95, 0.4)

    def test_evaluation_rejects_invalid_metrics_and_output_overlap(self) -> None:
        model = MagicMock()
        model.val.return_value.results_dict = {
            "metrics/precision(B)": 1.2,
            "metrics/recall(B)": 0.5,
            "metrics/mAP50(B)": 0.5,
            "metrics/mAP50-95(B)": 0.3,
        }
        yolo_class = MagicMock(return_value=model)
        with patch(
            "src.mosquito_detector.evaluate._load_yolo_class",
            return_value=yolo_class,
        ):
            with self.assertRaisesRegex(ValueError, "overlap dataset"):
                evaluate_model(
                    self.weights,
                    self.dataset_config,
                    output_dir=self.dataset_root / "val" / "plots",
                )
            with self.assertRaisesRegex(RuntimeError, "valid measured precision"):
                evaluate_model(
                    self.weights,
                    self.dataset_config,
                    output_dir=self.root / "bad-metrics",
                )

    def test_visualizers_reject_invalid_geometry_or_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            save_annotated_image(
                self.image_path,
                [],
                self.image_path,
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            save_evaluation_plot({"precision": 0.2}, self.root / "metrics.png")
        with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
            save_evaluation_plot(
                {
                    "precision": 0.2,
                    "recall": 0.3,
                    "map50": 0.4,
                    "map50_95": 1.1,
                },
                self.root / "metrics.png",
            )


if __name__ == "__main__":
    unittest.main()
