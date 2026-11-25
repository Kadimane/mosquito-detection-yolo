"""Unit tests for YOLO dataset validation."""

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.mosquito_detector.data.validate import validate_yolo_dataset


class ValidateYoloDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_image(self, name: str = "sample.jpg") -> Path:
        image = self.images_dir / name
        image.write_bytes(b"placeholder image")
        return image

    def write_label(self, content: str, name: str = "sample.txt") -> Path:
        label = self.labels_dir / name
        label.write_text(content, encoding="utf-8")
        return label

    def validate(self, num_classes: int = 2):
        with patch(
            "src.mosquito_detector.data.validate._inspect_image",
            return_value=None,
        ):
            return validate_yolo_dataset(
                self.images_dir,
                self.labels_dir,
                num_classes,
            )

    def test_valid_yolo_annotation_is_counted(self) -> None:
        self.write_image()
        self.write_label("1 0.5 0.5 0.25 0.4\n")

        report = self.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.image_count, 1)
        self.assertEqual(report.label_count, 1)
        self.assertEqual(report.matched_count, 1)
        self.assertEqual(report.annotation_count, 1)
        self.assertEqual(report.valid_annotation_count, 1)

    def test_image_and_label_matching_ignores_filename_case(self) -> None:
        self.write_image("Mosquito.JPG")
        self.write_label("1 0.5 0.5 0.25 0.4\n", "mosquito.txt")

        report = self.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.matched_count, 1)
        self.assertEqual(report.missing_label_count, 0)
        self.assertEqual(report.unmatched_label_count, 0)

    def test_case_insensitive_duplicate_labels_are_reported_as_ambiguous(self) -> None:
        self.write_image("mosquito.jpg")
        self.write_label("0 0.5 0.5 0.2 0.2\n", "mosquito.txt")
        self.write_label("0 0.5 0.5 0.2 0.2\n", "MOSQUITO.TXT")

        report = self.validate()
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.is_valid)
        self.assertIn("duplicate_label_filename", codes)
        self.assertIn("ambiguous_match", codes)
        self.assertEqual(report.matched_count, 0)
        self.assertEqual(report.unmatched_label_count, 0)

    def test_case_insensitive_duplicate_image_stems_are_ambiguous(self) -> None:
        self.write_image("mosquito.jpg")
        self.write_image("MOSQUITO.png")
        self.write_label("0 0.5 0.5 0.2 0.2\n", "Mosquito.txt")

        report = self.validate()
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.is_valid)
        self.assertIn("duplicate_image_stem", codes)
        self.assertIn("ambiguous_match", codes)
        self.assertEqual(report.matched_count, 0)

    def test_missing_and_unmatched_labels_are_reported(self) -> None:
        self.write_image("missing.jpg")
        self.write_label("0 0.5 0.5 0.2 0.2\n", "orphan.txt")

        report = self.validate()
        codes = {issue.code for issue in report.issues}

        self.assertEqual(report.missing_label_count, 1)
        self.assertEqual(report.unmatched_label_count, 1)
        self.assertIn("missing_label", codes)
        self.assertIn("unmatched_label", codes)

    def test_empty_label_is_reported_as_warning(self) -> None:
        self.write_image()
        self.write_label(" \n\t\n")

        report = self.validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.empty_label_count, 1)
        self.assertEqual(report.warning_count, 1)
        self.assertEqual(report.error_count, 0)

    def test_unreadable_image_is_reported(self) -> None:
        self.write_image()
        self.write_label("0 0.5 0.5 0.2 0.2\n")
        with patch(
            "src.mosquito_detector.data.validate._inspect_image",
            return_value="invalid image data",
        ):
            report = validate_yolo_dataset(
                self.images_dir,
                self.labels_dir,
                2,
            )

        self.assertFalse(report.is_valid)
        self.assertEqual(report.unreadable_image_count, 1)
        self.assertIn("invalid image data", str(report))

    @unittest.skipUnless(
        find_spec("PIL") is not None,
        "Pillow is required for the image-decoding integration test",
    )
    def test_corrupt_image_is_rejected_by_image_reader(self) -> None:
        self.write_image()
        self.write_label("0 0.5 0.5 0.2 0.2\n")

        report = validate_yolo_dataset(self.images_dir, self.labels_dir, 2)

        self.assertFalse(report.is_valid)
        self.assertEqual(report.unreadable_image_count, 1)

    def test_invalid_annotation_fields_and_values_are_reported(self) -> None:
        self.write_image()
        label = self.write_label(
            "0 0.5 0.5 0.2\n"
            "2 0.5 0.5 0.2 0.2\n"
            "0 NaN 0.5 0.2 0.2\n"
            "0 1.1 0.5 0 0.2\n"
            "0 0.5 0.5 0.2 nope\n"
        )

        report = self.validate()
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.is_valid)
        self.assertEqual(report.annotation_count, 5)
        self.assertEqual(report.valid_annotation_count, 0)
        self.assertTrue(
            {
                "invalid_field_count",
                "class_id_out_of_range",
                "non_finite_coordinate",
                "coordinate_out_of_range",
                "non_positive_width",
                "invalid_coordinate",
            }.issubset(codes)
        )
        self.assertTrue(all(issue.path == label for issue in report.issues))

    def test_boxes_extending_outside_image_boundaries_are_rejected(self) -> None:
        self.write_image()
        label = self.write_label(
            "0 0.1 0.8 0.4 0.5\n"
            "0 0.5 0.5 1 1\n"
        )

        report = self.validate()
        boundary_issues = [
            issue for issue in report.issues if issue.code == "box_out_of_bounds"
        ]

        self.assertFalse(report.is_valid)
        self.assertEqual(report.annotation_count, 2)
        self.assertEqual(report.valid_annotation_count, 1)
        self.assertEqual(len(boundary_issues), 2)
        self.assertTrue(all(issue.path == label for issue in boundary_issues))
        self.assertEqual({issue.line_number for issue in boundary_issues}, {1})
        self.assertTrue(any("horizontal" in issue.message for issue in boundary_issues))
        self.assertTrue(any("vertical" in issue.message for issue in boundary_issues))

    def test_invalid_num_classes_is_rejected(self) -> None:
        for num_classes in (0, -1, True, 2.5):
            with self.subTest(num_classes=num_classes):
                with self.assertRaises(ValueError):
                    validate_yolo_dataset(
                        self.images_dir,
                        self.labels_dir,
                        num_classes,  # type: ignore[arg-type]
                    )

    def test_summary_is_human_readable(self) -> None:
        self.write_image()
        self.write_label("0 0.5 0.5 0.2 0.2\n")

        summary = self.validate().summary()

        self.assertIn("YOLO dataset validation: VALID", summary)
        self.assertIn("matched pairs: 1", summary)
        self.assertIn("annotations: 1/1 valid", summary)


if __name__ == "__main__":
    unittest.main()
