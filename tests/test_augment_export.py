"""Synthetic and mocked tests for augmentation and ONNX export."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from src.mosquito_detector.data.augment import (
    AugmentationConfig,
    adjust_brightness_contrast,
    augment_image,
    horizontal_flip,
    rotate_image,
    scale_and_crop,
)
from src.mosquito_detector.data.preprocess import YoloAnnotation
from src.mosquito_detector.export import (
    ModelExportError,
    export_to_onnx,
)


class ImageAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (100, 100), (100, 120, 140))
        self.annotation = YoloAnnotation(2, 0.25, 0.5, 0.2, 0.4)

    def test_horizontal_flip_mirrors_box_without_mutating_source(self) -> None:
        original_pixel = self.image.getpixel((10, 50))

        flipped, annotations = horizontal_flip(self.image, [self.annotation])

        self.assertEqual(flipped.size, self.image.size)
        self.assertEqual(annotations, [YoloAnnotation(2, 0.75, 0.5, 0.2, 0.4)])
        self.assertEqual(self.image.getpixel((10, 50)), original_pixel)

    def test_rotation_transforms_box_and_clips_to_image(self) -> None:
        annotation = YoloAnnotation(0, 0.25, 0.5, 0.2, 0.4)

        rotated, annotations = rotate_image(
            self.image,
            [annotation],
            angle_degrees=90,
        )

        self.assertEqual(rotated.size, (100, 100))
        self.assertAlmostEqual(annotations[0].center_x, 0.5)
        self.assertAlmostEqual(annotations[0].center_y, 0.75)
        self.assertAlmostEqual(annotations[0].width, 0.4)
        self.assertAlmostEqual(annotations[0].height, 0.2)

    def test_rotation_drops_boxes_that_are_fully_outside_canvas(self) -> None:
        _, annotations = rotate_image(
            self.image,
            [YoloAnnotation(0, 0.01, 0.01, 0.02, 0.02)],
            45,
        )

        self.assertEqual(annotations, [])

    def test_scale_and_center_crop_transform_boxes(self) -> None:
        _, enlarged_annotations = scale_and_crop(
            self.image,
            [self.annotation],
            2,
        )
        self.assertAlmostEqual(enlarged_annotations[0].center_x, 0.1)
        self.assertAlmostEqual(enlarged_annotations[0].center_y, 0.5)
        self.assertAlmostEqual(enlarged_annotations[0].width, 0.2)
        self.assertAlmostEqual(enlarged_annotations[0].height, 0.8)

        _, reduced_annotations = scale_and_crop(
            self.image,
            [YoloAnnotation(0, 0.25, 0.5, 0.2, 0.4)],
            0.5,
        )
        self.assertAlmostEqual(reduced_annotations[0].center_x, 0.375)
        self.assertAlmostEqual(reduced_annotations[0].center_y, 0.5)
        self.assertAlmostEqual(reduced_annotations[0].width, 0.1)
        self.assertAlmostEqual(reduced_annotations[0].height, 0.2)

    def test_color_adjustment_does_not_change_annotation_geometry(self) -> None:
        adjusted = adjust_brightness_contrast(
            self.image,
            brightness_factor=0.5,
            contrast_factor=1.5,
        )

        self.assertEqual(adjusted.size, self.image.size)
        self.assertEqual(self.image.getpixel((0, 0)), (100, 120, 140))
        self.assertNotEqual(adjusted.getpixel((0, 0)), self.image.getpixel((0, 0)))

    def test_augment_config_applies_consistent_composition(self) -> None:
        result = augment_image(
            self.image,
            [self.annotation],
            AugmentationConfig(
                horizontal_flip=True,
                rotation_degrees=0,
                brightness_factor=0.8,
                contrast_factor=1,
                scale_factor=1,
            ),
        )

        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.annotations, (YoloAnnotation(2, 0.75, 0.5, 0.2, 0.4),))

    def test_invalid_augmentation_options_and_annotations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite number"):
            rotate_image(self.image, [], float("nan"))
        with self.assertRaisesRegex(ValueError, "positive"):
            scale_and_crop(self.image, [], 0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            adjust_brightness_contrast(self.image, brightness_factor=-1)
        with self.assertRaisesRegex(ValueError, "Invalid YOLO annotation"):
            horizontal_flip(
                self.image,
                [YoloAnnotation(0, 0.5, 0.5, 1.2, 0.2)],
            )

    def test_augmented_results_do_not_modify_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.png"
            self.image.save(path)
            original_bytes = path.read_bytes()

            augment_image(
                self.image,
                [self.annotation],
                AugmentationConfig(horizontal_flip=True, scale_factor=1.2),
            )

            self.assertEqual(path.read_bytes(), original_bytes)


class OnnxExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"mock weights")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_export_passes_options_and_moves_export_to_requested_path(self) -> None:
        generated = self.root / "best.onnx"
        target = self.root / "exports" / "mosquito.onnx"
        model = MagicMock()

        def fake_export(**kwargs):
            del kwargs
            generated.write_bytes(b"mock onnx")
            return str(generated)

        model.export.side_effect = fake_export
        yolo_class = MagicMock(return_value=model)
        original_weights = self.weights.read_bytes()

        with patch(
            "src.mosquito_detector.export._load_yolo_class",
            return_value=yolo_class,
        ):
            result = export_to_onnx(
                self.weights,
                target,
                image_size=320,
                dynamic=True,
                simplify=False,
                opset=13,
                half=False,
                device="cpu",
            )

        yolo_class.assert_called_once_with(str(self.weights))
        model.export.assert_called_once_with(
            format="onnx",
            imgsz=320,
            dynamic=True,
            simplify=False,
            half=False,
            device="cpu",
            opset=13,
        )
        self.assertEqual(result.onnx_path, target)
        self.assertEqual(target.read_bytes(), b"mock onnx")
        self.assertFalse(generated.exists())
        self.assertEqual(self.weights.read_bytes(), original_weights)

    def test_export_uses_default_path_and_supports_overwrite(self) -> None:
        existing_export = self.root / "best.onnx"
        existing_export.write_bytes(b"old")
        model = MagicMock()

        def fake_export(**kwargs):
            del kwargs
            existing_export.write_bytes(b"new")
            return str(existing_export)

        model.export.side_effect = fake_export
        with patch(
            "src.mosquito_detector.export._load_yolo_class",
            return_value=MagicMock(return_value=model),
        ):
            with self.assertRaises(FileExistsError):
                export_to_onnx(self.weights)
            result = export_to_onnx(self.weights, overwrite=True)

        self.assertEqual(result.onnx_path, existing_export)
        self.assertEqual(existing_export.read_bytes(), b"new")

    def test_export_validates_weights_options_and_destination_before_loading(self) -> None:
        loader = MagicMock()
        with patch("src.mosquito_detector.export._load_yolo_class", loader):
            with self.assertRaises(FileNotFoundError):
                export_to_onnx(self.root / "missing.pt")
            invalid_options = (
                {"image_size": 0},
                {"image_size": True},
                {"dynamic": 1},
                {"opset": 0},
                {"device": ""},
                {"output_path": self.root / "not-onnx.bin"},
            )
            for options in invalid_options:
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        export_to_onnx(self.weights, **options)
        loader.assert_not_called()

    def test_export_wraps_exporter_errors_and_missing_export_file(self) -> None:
        model = MagicMock()
        model.export.side_effect = RuntimeError("export failed")
        loader = MagicMock(return_value=MagicMock(return_value=model))

        with patch("src.mosquito_detector.export._load_yolo_class", loader):
            with self.assertLogs("src.mosquito_detector.export", level="ERROR"):
                with self.assertRaisesRegex(ModelExportError, "export failed"):
                    export_to_onnx(self.weights, self.root / "target.onnx")

            def missing_export(**kwargs):
                del kwargs
                return str(self.root / "missing.onnx")

            model.export.side_effect = missing_export
            with self.assertLogs("src.mosquito_detector.export", level="ERROR"):
                with self.assertRaisesRegex(ModelExportError, "was not found"):
                    export_to_onnx(
                        self.weights,
                        self.root / "target-2.onnx",
                    )


if __name__ == "__main__":
    unittest.main()
