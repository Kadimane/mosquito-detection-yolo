"""Unit tests for deterministic, non-destructive dataset splitting."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.mosquito_detector.data.split import split_yolo_dataset


class SplitYoloDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def add_sample(self, stem: str, *, label: bool = True) -> tuple[Path, Path | None]:
        image_path = self.images_dir / f"{stem}.png"
        Image.new("RGB", (4, 3), (10, 20, 30)).save(image_path)
        label_path = None
        if label:
            label_path = self.labels_dir / f"{stem}.txt"
            label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        return image_path, label_path

    def split(self, **kwargs):
        return split_yolo_dataset(self.images_dir, self.labels_dir, **kwargs)

    def test_default_ratios_assign_all_images_with_expected_counts(self) -> None:
        for index in range(10):
            self.add_sample(f"sample-{index}")

        result = self.split()

        self.assertEqual(len(result.train), 7)
        self.assertEqual(len(result.validation), 2)
        self.assertEqual(len(result.test), 1)
        self.assertEqual(
            len(result.train) + len(result.validation) + len(result.test),
            10,
        )
        self.assertIn("Train: 7 images (7 with annotations)", result.summary())

    def test_seed_produces_reproducible_assignments(self) -> None:
        for index in range(20):
            self.add_sample(f"sample-{index}")

        first = self.split(seed=123)
        second = self.split(seed=123)
        different = self.split(seed=456)

        def image_names(split):
            return tuple(
                tuple(item.image_path.name for item in items)
                for items in (split.train, split.validation, split.test)
            )

        self.assertEqual(image_names(first), image_names(second))
        self.assertNotEqual(image_names(first), image_names(different))

    def test_image_annotation_pairs_are_kept_and_missing_labels_reported(self) -> None:
        paired = self.add_sample("paired")
        missing, _ = self.add_sample("no-label", label=False)
        orphan_label = self.labels_dir / "orphan.txt"
        orphan_label.write_text("", encoding="utf-8")

        result = self.split(train_ratio=1, validation_ratio=0, test_ratio=0)

        assignments = result.train + result.validation + result.test
        self.assertEqual(len(assignments), 2)
        self.assertEqual(
            {item.image_path for item in assignments},
            {paired[0], missing},
        )
        self.assertEqual(
            {item.annotation_path for item in assignments if item.annotation_path},
            {paired[1]},
        )
        self.assertEqual(result.missing_annotations, (missing,))
        self.assertEqual(result.unmatched_annotations, (orphan_label,))
        self.assertIn("Images missing annotations: 1", result.summary())
        self.assertIn("Unmatched annotation files: 1", result.summary())

    def test_no_image_is_assigned_to_multiple_splits(self) -> None:
        for index in range(17):
            self.add_sample(f"image-{index}")

        result = self.split(train_ratio=0.6, validation_ratio=0.25, test_ratio=0.15)
        all_images = [
            item.image_path
            for items in (result.train, result.validation, result.test)
            for item in items
        ]

        self.assertEqual(len(all_images), len(set(all_images)))
        self.assertEqual(set(all_images), set(self.images_dir.glob("*.png")))

    def test_case_insensitive_matching_works(self) -> None:
        image_path, label_path = self.add_sample("Mosquito", label=False)
        assert label_path is None
        label_path = self.labels_dir / "mosquito.txt"
        label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

        result = self.split(train_ratio=1, validation_ratio=0, test_ratio=0)

        self.assertEqual(result.train[0].image_path, image_path)
        self.assertEqual(result.train[0].annotation_path, label_path)

    def test_duplicate_case_insensitive_stems_are_rejected(self) -> None:
        self.add_sample("mosquito")
        self.add_sample("MOSQUITO", label=False)

        with self.assertRaisesRegex(ValueError, "Ambiguous image match"):
            self.split()

    def test_invalid_ratios_are_rejected(self) -> None:
        for ratios in (
            (0.6, 0.2, 0.1),
            (-0.1, 0.6, 0.5),
            (1.1, 0, -0.1),
            (float("nan"), 0.2, 0.8),
            (float("inf"), 0, 0),
            (10**10000, 0, 0),
            (True, 0, 0),
        ):
            with self.subTest(ratios=ratios):
                with self.assertRaises(ValueError):
                    self.split(
                        train_ratio=ratios[0],
                        validation_ratio=ratios[1],
                        test_ratio=ratios[2],
                    )

    def test_non_integer_seed_is_rejected(self) -> None:
        for seed in (True, 1.5, "12"):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed must be an integer"):
                    self.split(seed=seed)

    def test_source_files_remain_unchanged(self) -> None:
        image_path, label_path = self.add_sample("source")
        assert label_path is not None
        original_image_bytes = image_path.read_bytes()
        original_label_bytes = label_path.read_bytes()

        self.split(train_ratio=1, validation_ratio=0, test_ratio=0)

        self.assertEqual(image_path.read_bytes(), original_image_bytes)
        self.assertEqual(label_path.read_bytes(), original_label_bytes)
        self.assertTrue(image_path.exists())
        self.assertTrue(label_path.exists())

    def test_empty_dataset_returns_empty_splits(self) -> None:
        result = self.split()

        self.assertEqual(result.train, ())
        self.assertEqual(result.validation, ())
        self.assertEqual(result.test, ())
        self.assertIn("Train: 0 images", result.summary())


if __name__ == "__main__":
    unittest.main()
