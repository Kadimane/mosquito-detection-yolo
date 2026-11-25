"""Unit tests for reusable dataset preprocessing utilities."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.mosquito_detector.data.preprocess import (
    LetterboxTransform,
    YoloAnnotation,
    letterbox_image,
    match_image_and_annotation_files,
    read_image,
    read_yolo_annotations,
    transform_yolo_annotations,
)


class PreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_read_image_returns_detached_rgb_image(self) -> None:
        path = self.root / "sample.png"
        Image.new("RGBA", (12, 8), (20, 40, 60, 128)).save(path)

        image = read_image(path)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (12, 8))
        self.assertEqual(image.getpixel((0, 0)), (20, 40, 60))

    def test_read_image_rejects_missing_and_invalid_files(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            read_image(self.root / "missing.png")

        invalid_path = self.root / "invalid.png"
        invalid_path.write_text("not an image", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid image file"):
            read_image(invalid_path)

    def test_read_yolo_annotations_parses_valid_rows_and_ignores_blank_lines(self) -> None:
        path = self.root / "sample.txt"
        path.write_text(
            "\n  1 0.5 0.4 0.2 0.3  \n0 0.2 0.3 0.1 0.15\n",
            encoding="utf-8",
        )

        annotations = read_yolo_annotations(path, num_classes=2)

        self.assertEqual(
            annotations,
            [
                YoloAnnotation(1, 0.5, 0.4, 0.2, 0.3),
                YoloAnnotation(0, 0.2, 0.3, 0.1, 0.15),
            ],
        )

    def test_empty_annotation_file_is_explicitly_returned_as_no_boxes(self) -> None:
        path = self.root / "background.txt"
        path.write_text(" \n\t\n", encoding="utf-8")

        self.assertEqual(read_yolo_annotations(path), [])

    def test_missing_annotation_file_is_not_treated_as_empty(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            read_yolo_annotations(self.root / "missing.txt")

    def test_malformed_annotation_has_file_and_line_in_error(self) -> None:
        path = self.root / "bad.txt"
        path.write_text(
            "0 0.5 0.5 0.2 0.2\n0 NaN 0.5 0.2 0.2\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, r"bad\.txt:2"):
            read_yolo_annotations(path)

    def test_annotation_parser_rejects_invalid_class_and_dimensions(self) -> None:
        path = self.root / "bad.txt"
        path.write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "class ID"):
            read_yolo_annotations(path, num_classes=2)

        path.write_text("0 0.5 0.5 0 0.2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "width"):
            read_yolo_annotations(path)

    def test_match_uses_case_insensitive_stems_and_reports_missing_or_orphaned(self) -> None:
        images = [
            self.root / "Mosquito.JPG",
            self.root / "missing.png",
        ]
        labels = [
            self.root / "mosquito.txt",
            self.root / "orphan.TXT",
        ]

        result = match_image_and_annotation_files(images, labels)

        self.assertEqual(result.matches, ((images[0], labels[0]),))
        self.assertEqual(result.missing_annotations, (images[1],))
        self.assertEqual(result.unmatched_annotations, (labels[1],))

    def test_match_rejects_case_insensitive_duplicate_stems(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ambiguous image match"):
            match_image_and_annotation_files(
                [self.root / "mosquito.jpg", self.root / "MOSQUITO.png"],
                [self.root / "mosquito.txt"],
            )

        with self.assertRaisesRegex(ValueError, "Ambiguous annotation match"):
            match_image_and_annotation_files(
                [self.root / "mosquito.jpg"],
                [self.root / "mosquito.txt", self.root / "MOSQUITO.TXT"],
            )

    def test_letterbox_preserves_aspect_ratio_and_centers_padding(self) -> None:
        original = Image.new("RGB", (100, 50), (255, 0, 0))

        resized, transform = letterbox_image(original, (100, 100))

        self.assertEqual(resized.size, (100, 100))
        self.assertEqual(transform.resized_width, 100)
        self.assertEqual(transform.resized_height, 50)
        self.assertEqual((transform.pad_left, transform.pad_top), (0, 25))
        self.assertEqual(resized.getpixel((50, 0)), (114, 114, 114))
        self.assertEqual(resized.getpixel((50, 25)), (255, 0, 0))
        self.assertEqual(original.size, (100, 50))

    def test_transform_updates_normalized_box_coordinates_for_padding(self) -> None:
        _, transform = letterbox_image(Image.new("RGB", (100, 50)), (100, 100))
        annotation = YoloAnnotation(0, 0.5, 0.25, 0.4, 0.4)

        result = transform_yolo_annotations([annotation], transform)

        self.assertEqual(result[0].class_id, 0)
        self.assertAlmostEqual(result[0].center_x, 0.5)
        self.assertAlmostEqual(result[0].center_y, 0.375)
        self.assertAlmostEqual(result[0].width, 0.4)
        self.assertAlmostEqual(result[0].height, 0.2)

    def test_transform_keeps_extreme_aspect_ratio_boxes_within_bounds(self) -> None:
        _, transform = letterbox_image(Image.new("RGB", (1, 1000)), (3, 7))

        result = transform_yolo_annotations(
            [YoloAnnotation(0, 0.5, 0.5, 1, 1)],
            transform,
        )[0]

        self.assertAlmostEqual(result.center_x - result.width / 2, 1 / 3)
        self.assertAlmostEqual(result.center_x + result.width / 2, 2 / 3)
        self.assertAlmostEqual(result.center_y - result.height / 2, 0)
        self.assertAlmostEqual(result.center_y + result.height / 2, 1)

    def test_transform_rejects_padding_that_places_boxes_outside_target(self) -> None:
        invalid_transform = LetterboxTransform(
            100,
            50,
            100,
            100,
            100,
            50,
            100,
            25,
        )

        with self.assertRaisesRegex(ValueError, "do not fit"):
            transform_yolo_annotations(
                [YoloAnnotation(0, 0.5, 0.5, 0.2, 0.2)],
                invalid_transform,
            )

    def test_transform_rejects_non_integer_dimensions(self) -> None:
        invalid_transform = LetterboxTransform(
            100,
            50,
            100,
            100,
            100,
            50,
            0.5,  # type: ignore[arg-type]
            25,
        )

        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            transform_yolo_annotations([], invalid_transform)

    def test_letterbox_rejects_invalid_dimensions(self) -> None:
        image = Image.new("RGB", (10, 10))
        for size in ((0, 10), (10, -1), (10.5, 10), (True, 10)):
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    letterbox_image(image, size)  # type: ignore[arg-type]

    def test_readers_do_not_modify_source_files(self) -> None:
        image_path = self.root / "sample.png"
        annotation_path = self.root / "sample.txt"
        Image.new("RGB", (4, 3), (10, 20, 30)).save(image_path)
        annotation_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        original_image_bytes = image_path.read_bytes()
        original_annotation_bytes = annotation_path.read_bytes()

        read_image(image_path)
        read_yolo_annotations(annotation_path)

        self.assertEqual(image_path.read_bytes(), original_image_bytes)
        self.assertEqual(annotation_path.read_bytes(), original_annotation_bytes)

    def test_transform_rejects_invalid_annotation(self) -> None:
        _, transform = letterbox_image(Image.new("RGB", (10, 10)), (10, 10))

        with self.assertRaisesRegex(ValueError, "Invalid YOLO annotation"):
            transform_yolo_annotations(
                [YoloAnnotation(0, 0.5, 0.5, 1.2, 0.2)],
                transform,
            )


if __name__ == "__main__":
    unittest.main()
