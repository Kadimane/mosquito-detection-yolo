"""Validation utilities for YOLO object-detection datasets."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal


_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ValidationIssue:
    """A single actionable dataset validation finding."""

    severity: Literal["error", "warning"]
    code: str
    message: str
    path: Path
    line_number: int | None = None

    def __str__(self) -> str:
        """Return the issue in a concise, location-aware form."""
        location = str(self.path)
        if self.line_number is not None:
            location = f"{location}:{self.line_number}"
        return f"{self.severity.upper()} [{self.code}] {location}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """Counts and findings produced while validating a YOLO dataset."""

    image_count: int
    label_count: int
    matched_count: int
    missing_label_count: int
    unmatched_label_count: int
    empty_label_count: int
    unreadable_image_count: int
    annotation_count: int
    valid_annotation_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Whether the dataset has no validation errors."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Number of error-severity findings."""
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        """Number of warning-severity findings."""
        return sum(issue.severity == "warning" for issue in self.issues)

    def summary(self) -> str:
        """Return a human-readable dataset validation summary."""
        status = "VALID" if self.is_valid else "INVALID"
        return "\n".join(
            (
                f"YOLO dataset validation: {status}",
                f"Images: {self.image_count}; labels: {self.label_count}; "
                f"matched pairs: {self.matched_count}",
                f"Missing labels: {self.missing_label_count}; "
                f"unmatched labels: {self.unmatched_label_count}; "
                f"empty labels: {self.empty_label_count}",
                f"Unreadable images: {self.unreadable_image_count}; "
                f"annotations: {self.valid_annotation_count}/"
                f"{self.annotation_count} valid",
                f"Errors: {self.error_count}; warnings: {self.warning_count}",
            )
        )

    def __str__(self) -> str:
        """Return the summary followed by individual findings."""
        details = "\n".join(str(issue) for issue in self.issues)
        return f"{self.summary()}\n{details}" if details else self.summary()


def _inspect_image(path: Path) -> str | None:
    """Return an error description if Pillow cannot verify an image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Image validation requires Pillow; install project dependencies "
            "with `python -m pip install -r requirements.txt`."
        ) from exc

    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError) as exc:
        return str(exc) or type(exc).__name__
    return None


def _validate_annotation_line(
    fields: list[str],
    path: Path,
    line_number: int,
    num_classes: int | None,
) -> tuple[bool, list[ValidationIssue]]:
    """Validate one non-empty YOLO label row."""
    issues: list[ValidationIssue] = []
    if len(fields) != 5:
        return False, [
            ValidationIssue(
                "error",
                "invalid_field_count",
                f"expected 5 whitespace-separated values, found {len(fields)}",
                path,
                line_number,
            )
        ]

    class_id_text, *coordinate_text = fields
    try:
        class_id = int(class_id_text)
    except ValueError:
        class_id = -1
        issues.append(
            ValidationIssue(
                "error",
                "invalid_class_id",
                f"class ID {class_id_text!r} must be an integer",
                path,
                line_number,
            )
        )
    else:
        if class_id < 0 or (
            num_classes is not None and class_id >= num_classes
        ):
            range_description = (
                f"0..{num_classes - 1}" if num_classes is not None else "0 or greater"
            )
            issues.append(
                ValidationIssue(
                    "error",
                    "class_id_out_of_range",
                    f"class ID {class_id} is outside the valid range "
                    f"{range_description}",
                    path,
                    line_number,
                )
            )

    coordinates: list[float | None] = []
    for field_name, value in zip(
        ("center_x", "center_y", "width", "height"),
        coordinate_text,
    ):
        try:
            coordinate = float(value)
        except ValueError:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_coordinate",
                    f"{field_name} value {value!r} is not numeric",
                    path,
                    line_number,
                )
            )
            coordinates.append(None)
            continue

        if not math.isfinite(coordinate):
            issues.append(
                ValidationIssue(
                    "error",
                    "non_finite_coordinate",
                    f"{field_name} value must be finite, found {value!r}",
                    path,
                    line_number,
                )
            )
            coordinates.append(None)
        else:
            coordinates.append(coordinate)
            if not 0 <= coordinate <= 1:
                issues.append(
                    ValidationIssue(
                        "error",
                        "coordinate_out_of_range",
                        f"{field_name} must be normalized to [0, 1], "
                        f"found {coordinate}",
                        path,
                        line_number,
                    )
                )

    if all(coordinate is not None for coordinate in coordinates):
        center_x, center_y, width, height = coordinates
        assert center_x is not None
        assert center_y is not None
        assert width is not None
        assert height is not None
        if width <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "non_positive_width",
                    f"box width must be greater than 0, found {width}",
                    path,
                    line_number,
                )
            )
        else:
            left = center_x - width / 2
            right = center_x + width / 2
            if left < 0 or right > 1:
                issues.append(
                    ValidationIssue(
                        "error",
                        "box_out_of_bounds",
                        f"horizontal box edges [{left}, {right}] must remain "
                        "within [0, 1]",
                        path,
                        line_number,
                    )
                )
        if height <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "non_positive_height",
                    f"box height must be greater than 0, found {height}",
                    path,
                    line_number,
                )
            )
        else:
            top = center_y - height / 2
            bottom = center_y + height / 2
            if top < 0 or bottom > 1:
                issues.append(
                    ValidationIssue(
                        "error",
                        "box_out_of_bounds",
                        f"vertical box edges [{top}, {bottom}] must remain "
                        "within [0, 1]",
                        path,
                        line_number,
                    )
                )

    return not issues, issues


def validate_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    num_classes: int,
) -> ValidationReport:
    """Validate image/label pairing and YOLO detection annotations.

    Each supported image is matched to a `.txt` label file with the same
    filename stem, ignoring case. Non-empty annotation rows must use the YOLO
    detection format `class_id center_x center_y width height`. Empty label
    files are reported as warnings because they can intentionally represent
    background images.

    Args:
        images_dir: Directory containing the dataset images.
        labels_dir: Directory containing YOLO `.txt` annotation files.
        num_classes: Number of classes; valid class IDs are in
            ``range(num_classes)``.

    Returns:
        A report containing aggregate counts and location-specific findings.

    Raises:
        ValueError: If ``num_classes`` is not a positive integer.
        ImportError: If Pillow is unavailable when an image is inspected.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise ValueError("num_classes must be a positive integer")
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    image_root = Path(images_dir)
    label_root = Path(labels_dir)
    issues: list[ValidationIssue] = []

    image_paths: list[Path] = []
    label_paths: list[Path] = []
    for directory, extensions, result, code, description in (
        (
            image_root,
            _IMAGE_EXTENSIONS,
            image_paths,
            "invalid_image_directory",
            "image",
        ),
        (
            label_root,
            {".txt"},
            label_paths,
            "invalid_label_directory",
            "label",
        ),
    ):
        if not directory.is_dir():
            issues.append(
                ValidationIssue(
                    "error",
                    code,
                    f"{description} directory does not exist or is not a directory",
                    directory,
                )
            )
            continue
        try:
            result.extend(
                sorted(
                    (
                        path
                        for path in directory.iterdir()
                        if path.is_file() and path.suffix.lower() in extensions
                    ),
                    key=lambda path: path.name.lower(),
                )
            )
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    "error",
                    code,
                    f"cannot list {description} directory: {exc}",
                    directory,
                )
            )

    images_by_stem: dict[str, list[Path]] = {}
    for image_path in image_paths:
        images_by_stem.setdefault(image_path.stem.casefold(), []).append(image_path)

    labels_by_stem: dict[str, list[Path]] = {}
    labels_by_name: dict[str, list[Path]] = {}
    for label_path in label_paths:
        labels_by_stem.setdefault(label_path.stem.casefold(), []).append(label_path)
        labels_by_name.setdefault(label_path.name.casefold(), []).append(label_path)

    for stem, paths in images_by_stem.items():
        if len(paths) > 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_image_stem",
                    f"multiple images share the case-insensitive stem "
                    f"{paths[0].stem!r}; label matching is ambiguous",
                    paths[0],
                )
            )

    for paths in labels_by_name.values():
        if len(paths) > 1:
            filenames = ", ".join(path.name for path in paths)
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_label_filename",
                    f"label filenames are duplicates ignoring case: {filenames}",
                    paths[0],
                )
            )

    missing_labels = 0
    unmatched_labels = 0
    matched_count = 0
    empty_labels = 0
    unreadable_images = 0
    annotation_count = 0
    valid_annotation_count = 0

    for image_path in image_paths:
        try:
            image_error = _inspect_image(image_path)
        except ImportError:
            raise
        if image_error is not None:
            unreadable_images += 1
            issues.append(
                ValidationIssue(
                    "error",
                    "unreadable_image",
                    f"image could not be verified: {image_error}",
                    image_path,
                )
            )

        stem_key = image_path.stem.casefold()
        matching_labels = labels_by_stem.get(stem_key, [])
        if not matching_labels:
            missing_labels += 1
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_label",
                    f"no .txt label file matches image stem {image_path.stem!r} "
                    "ignoring case",
                    image_path,
                )
            )
            continue

        matching_images = images_by_stem[stem_key]
        if len(matching_labels) > 1 or len(matching_images) > 1:
            filenames = ", ".join(path.name for path in matching_labels)
            image_names = ", ".join(path.name for path in matching_images)
            issues.append(
                ValidationIssue(
                    "error",
                    "ambiguous_match",
                    f"case-insensitive stem {image_path.stem!r} has "
                    f"{len(matching_images)} images ({image_names}) and "
                    f"{len(matching_labels)} labels ({filenames})",
                    image_path,
                )
            )
        else:
            matched_count += 1

    image_stems = set(images_by_stem)
    for label_path in label_paths:
        stem_key = label_path.stem.casefold()
        if stem_key not in image_stems:
            unmatched_labels += 1
            issues.append(
                ValidationIssue(
                    "error",
                    "unmatched_label",
                    f"no image file matches label stem {label_path.stem!r} "
                    "ignoring case",
                    label_path,
                )
            )
            continue

        try:
            label_text = label_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            issues.append(
                ValidationIssue(
                    "error",
                    "unreadable_label",
                    f"label file could not be read as UTF-8: {exc}",
                    label_path,
                )
            )
            continue

        rows = [
            (line_number, line.split())
            for line_number, line in enumerate(label_text.splitlines(), start=1)
            if line.strip()
        ]
        if not rows:
            empty_labels += 1
            issues.append(
                ValidationIssue(
                    "warning",
                    "empty_label",
                    "label file is empty; this is valid only for an intentional background image",
                    label_path,
                )
            )
            continue

        for line_number, fields in rows:
            annotation_count += 1
            valid, row_issues = _validate_annotation_line(
                fields,
                label_path,
                line_number,
                num_classes,
            )
            if valid:
                valid_annotation_count += 1
            issues.extend(row_issues)

    return ValidationReport(
        image_count=len(image_paths),
        label_count=len(label_paths),
        matched_count=matched_count,
        missing_label_count=missing_labels,
        unmatched_label_count=unmatched_labels,
        empty_label_count=empty_labels,
        unreadable_image_count=unreadable_images,
        annotation_count=annotation_count,
        valid_annotation_count=valid_annotation_count,
        issues=tuple(issues),
    )
