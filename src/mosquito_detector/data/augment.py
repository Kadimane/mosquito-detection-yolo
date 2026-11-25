"""Image augmentation helpers that keep YOLO detection boxes aligned."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from .preprocess import YoloAnnotation
from .validate import _validate_annotation_line

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class AugmentationConfig:
    """Options for applying deterministic augmentations to one image."""

    horizontal_flip: bool = False
    rotation_degrees: float = 0.0
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    scale_factor: float = 1.0


@dataclass(frozen=True)
class AugmentedSample:
    """Augmented image and its corresponding YOLO annotations."""

    image: Image.Image
    annotations: tuple[YoloAnnotation, ...]


def _validate_annotations(
    annotations: Iterable[YoloAnnotation],
) -> list[YoloAnnotation]:
    """Validate and materialize normalized detection annotations."""
    validated = list(annotations)
    for index, annotation in enumerate(validated, start=1):
        if not isinstance(annotation, YoloAnnotation):
            raise TypeError("annotations must contain YoloAnnotation values")
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
    return validated


def _map_boxes(
    annotations: Iterable[YoloAnnotation],
    image_width: int,
    image_height: int,
    transform: Callable[[float, float], tuple[float, float]],
) -> list[YoloAnnotation]:
    """Transform boxes, clip to image edges, and omit fully cropped boxes."""
    results: list[YoloAnnotation] = []
    for annotation in annotations:
        x1 = (annotation.center_x - annotation.width / 2) * image_width
        y1 = (annotation.center_y - annotation.height / 2) * image_height
        x2 = (annotation.center_x + annotation.width / 2) * image_width
        y2 = (annotation.center_y + annotation.height / 2) * image_height
        corners = [
            transform(x1, y1),
            transform(x1, y2),
            transform(x2, y1),
            transform(x2, y2),
        ]
        left = max(0.0, min(point[0] for point in corners))
        top = max(0.0, min(point[1] for point in corners))
        right = min(float(image_width), max(point[0] for point in corners))
        bottom = min(float(image_height), max(point[1] for point in corners))
        if right <= left or bottom <= top:
            continue
        results.append(
            YoloAnnotation(
                annotation.class_id,
                (left + right) / (2 * image_width),
                (top + bottom) / (2 * image_height),
                (right - left) / image_width,
                (bottom - top) / image_height,
            )
        )
    return results


def _require_image(image: Image.Image) -> None:
    """Raise a clear error unless image is a valid Pillow image."""
    from PIL import Image as PillowImage

    if not isinstance(image, PillowImage.Image):
        raise TypeError("image must be a Pillow Image instance")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("image dimensions must be positive")


def horizontal_flip(
    image: Image.Image,
    annotations: Iterable[YoloAnnotation],
) -> tuple[Image.Image, list[YoloAnnotation]]:
    """Flip an image horizontally and reflect normalized box centers."""
    from PIL import Image as PillowImage

    _require_image(image)
    boxes = _validate_annotations(annotations)
    flipped = image.transpose(PillowImage.Transpose.FLIP_LEFT_RIGHT)
    return flipped, [
        YoloAnnotation(
            box.class_id,
            1.0 - box.center_x,
            box.center_y,
            box.width,
            box.height,
        )
        for box in boxes
    ]


def rotate_image(
    image: Image.Image,
    annotations: Iterable[YoloAnnotation],
    angle_degrees: float,
) -> tuple[Image.Image, list[YoloAnnotation]]:
    """Rotate in place around the image center, clipping boxes to the canvas.

    Positive angles rotate counter-clockwise. The canvas size remains unchanged;
    boxes entirely outside the canvas after rotation are omitted.
    """
    from PIL import Image

    _require_image(image)
    if (
        isinstance(angle_degrees, bool)
        or not isinstance(angle_degrees, (int, float))
        or not math.isfinite(angle_degrees)
    ):
        raise ValueError("angle_degrees must be a finite number")
    boxes = _validate_annotations(annotations)
    radians = math.radians(float(angle_degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    center_x = image.width / 2
    center_y = image.height / 2

    def rotate(x: float, y: float) -> tuple[float, float]:
        relative_x = x - center_x
        relative_y = y - center_y
        return (
            cosine * relative_x + sine * relative_y + center_x,
            -sine * relative_x + cosine * relative_y + center_y,
        )

    rotated = image.rotate(
        float(angle_degrees),
        resample=Image.Resampling.BILINEAR,
        expand=False,
    )
    return rotated, _map_boxes(boxes, image.width, image.height, rotate)


def adjust_brightness_contrast(
    image: Image.Image,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
) -> Image.Image:
    """Adjust brightness and contrast without changing annotations or geometry."""
    from PIL import ImageEnhance

    _require_image(image)
    for name, factor in (
        ("brightness_factor", brightness_factor),
        ("contrast_factor", contrast_factor),
    ):
        if (
            isinstance(factor, bool)
            or not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or factor < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
    adjusted = ImageEnhance.Brightness(image).enhance(float(brightness_factor))
    return ImageEnhance.Contrast(adjusted).enhance(float(contrast_factor))


def scale_and_crop(
    image: Image.Image,
    annotations: Iterable[YoloAnnotation],
    scale_factor: float,
) -> tuple[Image.Image, list[YoloAnnotation]]:
    """Scale around the center and crop/pad to the original canvas dimensions.

    Scales above one crop at the edges; scales below one add black padding.
    Boxes are clipped to the output canvas, and boxes wholly outside it are
    omitted. Neither the source image nor its annotations are modified.
    """
    from PIL import Image

    _require_image(image)
    if (
        isinstance(scale_factor, bool)
        or not isinstance(scale_factor, (int, float))
        or not math.isfinite(scale_factor)
        or scale_factor <= 0
    ):
        raise ValueError("scale_factor must be a finite positive number")
    boxes = _validate_annotations(annotations)
    scaled_width = max(1, round(image.width * scale_factor))
    scaled_height = max(1, round(image.height * scale_factor))
    scaled_image = image.convert("RGB").resize(
        (scaled_width, scaled_height),
        Image.Resampling.BILINEAR,
    )
    offset_x = (image.width - scaled_width) // 2
    offset_y = (image.height - scaled_height) // 2
    canvas = Image.new("RGB", image.size, (0, 0, 0))
    canvas.paste(scaled_image, (offset_x, offset_y))
    scale_x = scaled_width / image.width
    scale_y = scaled_height / image.height

    def scale_and_translate(x: float, y: float) -> tuple[float, float]:
        return x * scale_x + offset_x, y * scale_y + offset_y

    transformed = _map_boxes(boxes, image.width, image.height, scale_and_translate)
    return canvas, transformed


def augment_image(
    image: Image.Image,
    annotations: Iterable[YoloAnnotation],
    config: AugmentationConfig = AugmentationConfig(),
) -> AugmentedSample:
    """Apply configured flips, rotation, scaling/cropping and color changes."""
    _require_image(image)
    if not isinstance(config, AugmentationConfig):
        raise TypeError("config must be an AugmentationConfig")
    if not isinstance(config.horizontal_flip, bool):
        raise ValueError("horizontal_flip must be a boolean")
    boxes = _validate_annotations(annotations)
    augmented = image.copy()
    if config.horizontal_flip:
        augmented, boxes = horizontal_flip(augmented, boxes)
    if config.rotation_degrees != 0:
        augmented, boxes = rotate_image(
            augmented,
            boxes,
            config.rotation_degrees,
        )
    if config.scale_factor != 1:
        augmented, boxes = scale_and_crop(
            augmented,
            boxes,
            config.scale_factor,
        )
    augmented = adjust_brightness_contrast(
        augmented,
        config.brightness_factor,
        config.contrast_factor,
    )
    return AugmentedSample(augmented, tuple(boxes))
