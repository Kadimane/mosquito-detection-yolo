"""Reusable image and YOLO annotation preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from .validate import _IMAGE_EXTENSIONS, _inspect_image, _validate_annotation_line

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class YoloAnnotation:
    """One detection box in normalized YOLO format."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class FileMatchResult:
    """Case-insensitive image/label matches, including files without partners."""

    matches: tuple[tuple[Path, Path], ...]
    missing_annotations: tuple[Path, ...]
    unmatched_annotations: tuple[Path, ...]


@dataclass(frozen=True)
class LetterboxTransform:
    """Image geometry needed to map normalized boxes into a letterboxed image."""

    original_width: int
    original_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int


def read_image(path: str | Path) -> Image.Image:
    """Read and validate an image, returning a detached RGB Pillow image.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file is not a readable image.
        ImportError: If Pillow is not installed.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image_error = _inspect_image(image_path)
    if image_error is not None:
        raise ValueError(f"Invalid image file {image_path}: {image_error}")

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            image.load()
            return image.convert("RGB").copy()
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid image file {image_path}: {exc}") from exc


def read_yolo_annotations(
    path: str | Path,
    num_classes: int | None = None,
) -> list[YoloAnnotation]:
    """Read YOLO detection rows, validating normalized coordinates.

    Empty annotation files return an empty list, representing an image with no
    objects. A missing file raises ``FileNotFoundError`` rather than being
    treated as an empty annotation file. When given, ``num_classes`` also
    constrains class IDs to ``range(num_classes)``.

    Raises:
        FileNotFoundError: If the annotation file does not exist.
        ValueError: If the file encoding, class count, or any annotation is invalid.
    """
    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {label_path}")
    if num_classes is not None and (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise ValueError("num_classes must be a positive integer")

    try:
        contents = label_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"Cannot read annotation file {label_path} as UTF-8: {exc}"
        ) from exc

    annotations: list[YoloAnnotation] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue

        valid, issues = _validate_annotation_line(
            fields,
            label_path,
            line_number,
            num_classes,
        )
        if not valid:
            details = "; ".join(issue.message for issue in issues)
            raise ValueError(
                f"Invalid YOLO annotation in {label_path}:{line_number}: {details}"
            )

        class_id, center_x, center_y, width, height = fields
        annotations.append(
            YoloAnnotation(
                int(class_id),
                float(center_x),
                float(center_y),
                float(width),
                float(height),
            )
        )
    return annotations


def match_image_and_annotation_files(
    image_paths: Iterable[str | Path],
    annotation_paths: Iterable[str | Path],
) -> FileMatchResult:
    """Match images to ``.txt`` annotations by case-insensitive filename stem.

    Images without labels and labels without images are returned explicitly.
    Duplicate case-insensitive stems raise ``ValueError`` because they make a
    one-to-one pairing ambiguous.
    """
    images = tuple(Path(path) for path in image_paths)
    annotations = tuple(Path(path) for path in annotation_paths)

    image_by_stem: dict[str, Path] = {}
    for image_path in images:
        stem = image_path.stem.casefold()
        if stem in image_by_stem:
            raise ValueError(
                f"Ambiguous image match: {image_by_stem[stem]} and {image_path} "
                f"share case-insensitive stem {stem!r}"
            )
        image_by_stem[stem] = image_path

    annotation_by_stem: dict[str, Path] = {}
    for annotation_path in annotations:
        stem = annotation_path.stem.casefold()
        if stem in annotation_by_stem:
            raise ValueError(
                "Ambiguous annotation match: "
                f"{annotation_by_stem[stem]} and {annotation_path} share "
                f"case-insensitive stem {stem!r}"
            )
        annotation_by_stem[stem] = annotation_path

    matches: list[tuple[Path, Path]] = []
    missing_annotations: list[Path] = []
    for image_path in images:
        label_path = annotation_by_stem.get(image_path.stem.casefold())
        if label_path is None:
            missing_annotations.append(image_path)
        else:
            matches.append((image_path, label_path))

    image_stems = set(image_by_stem)
    unmatched_annotations = tuple(
        annotation_path
        for annotation_path in annotations
        if annotation_path.stem.casefold() not in image_stems
    )
    return FileMatchResult(
        tuple(matches),
        tuple(missing_annotations),
        unmatched_annotations,
    )


def list_image_files(directory: str | Path) -> tuple[Path, ...]:
    """List supported image files in a directory in stable name order."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(
            f"Image directory does not exist or is not a directory: {root}"
        )
    try:
        return tuple(
            sorted(
                (
                    path
                    for path in root.iterdir()
                    if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
                ),
                key=lambda path: path.name.casefold(),
            )
        )
    except OSError as exc:
        raise OSError(f"Cannot list image directory {root}: {exc}") from exc


def list_annotation_files(directory: str | Path) -> tuple[Path, ...]:
    """List ``.txt`` annotation files in a directory in stable name order."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(
            f"Annotation directory does not exist or is not a directory: {root}"
        )
    try:
        return tuple(
            sorted(
                (
                    path
                    for path in root.iterdir()
                    if path.is_file() and path.suffix.lower() == ".txt"
                ),
                key=lambda path: path.name.casefold(),
            )
        )
    except OSError as exc:
        raise OSError(f"Cannot list annotation directory {root}: {exc}") from exc


def letterbox_image(
    image: Image.Image,
    target_size: tuple[int, int],
    fill_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, LetterboxTransform]:
    """Resize a Pillow image to ``(width, height)`` with centered padding.

    Returns the padded RGB image and a transform usable with
    :func:`transform_yolo_annotations`. Original image objects are not modified.
    """
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow Image instance")
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        raise ValueError("target_size must contain (width, height)")
    target_width, target_height = target_size
    if any(
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
        for dimension in target_size
    ):
        raise ValueError("target dimensions must be positive integers")
    original_width, original_height = image.size
    if original_width <= 0 or original_height <= 0:
        raise ValueError("image dimensions must be positive")

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, min(target_width, round(original_width * scale)))
    resized_height = max(1, min(target_height, round(original_height * scale)))
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2

    resized_image = image.convert("RGB").resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    letterboxed = Image.new("RGB", (target_width, target_height), fill_color)
    letterboxed.paste(resized_image, (pad_left, pad_top))
    transform = LetterboxTransform(
        original_width,
        original_height,
        target_width,
        target_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top,
    )
    return letterboxed, transform


def transform_yolo_annotations(
    annotations: Iterable[YoloAnnotation],
    transform: LetterboxTransform,
) -> list[YoloAnnotation]:
    """Map normalized YOLO boxes through a letterbox resize and padding."""
    if not isinstance(transform, LetterboxTransform):
        raise ValueError("transform must be a LetterboxTransform")

    dimensions = (
        transform.original_width,
        transform.original_height,
        transform.target_width,
        transform.target_height,
        transform.resized_width,
        transform.resized_height,
    )
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in dimensions
    ):
        raise ValueError("image dimensions in transform must be positive integers")
    if any(
        isinstance(padding, bool) or not isinstance(padding, int) or padding < 0
        for padding in (transform.pad_left, transform.pad_top)
    ):
        raise ValueError("transform padding must be non-negative integers")
    if (
        transform.resized_width > transform.target_width
        or transform.resized_height > transform.target_height
        or transform.pad_left
        != (transform.target_width - transform.resized_width) // 2
        or transform.pad_top
        != (transform.target_height - transform.resized_height) // 2
    ):
        raise ValueError("transform resize and padding do not fit the target image")

    scale_x = transform.resized_width / transform.original_width
    scale_y = transform.resized_height / transform.original_height
    transformed: list[YoloAnnotation] = []
    for index, annotation in enumerate(annotations, start=1):
        fields = [
            str(annotation.class_id),
            str(annotation.center_x),
            str(annotation.center_y),
            str(annotation.width),
            str(annotation.height),
        ]
        valid, issues = _validate_annotation_line(
            fields,
            Path("<annotation>"),
            index,
            None,
        )
        if not valid:
            details = "; ".join(issue.message for issue in issues)
            raise ValueError(f"Invalid YOLO annotation at index {index}: {details}")

        class_id = int(fields[0])
        center_x_original, center_y_original, width_original, height_original = (
            float(value) for value in fields[1:]
        )
        center_x = (
            center_x_original * transform.original_width * scale_x
            + transform.pad_left
        ) / transform.target_width
        center_y = (
            center_y_original * transform.original_height * scale_y
            + transform.pad_top
        ) / transform.target_height
        width = (
            width_original
            * transform.original_width
            * scale_x
            / transform.target_width
        )
        height = (
            height_original
            * transform.original_height
            * scale_y
            / transform.target_height
        )
        transformed.append(
            YoloAnnotation(
                class_id,
                center_x,
                center_y,
                width,
                height,
            )
        )
    return transformed
