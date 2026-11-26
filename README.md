# Mosquito Detection with YOLO

This repository contains reusable dataset utilities and an Ultralytics YOLO
workflow for object detection. It includes a Colab notebook for reproducing a
MosquitoFusion training run; dataset files and trained weights are not included.

> **Results status:** Colab has not yet downloaded the dataset or started
> training. No project-specific mAP score, trained checkpoint, or prediction
> image is claimed here. The paper's reported score below is a reference
> result, not a result from this repository.

## Architecture

```text
configs/
  train.yaml                 Training options
  data.yaml                  YOLO dataset paths and class names
src/mosquito_detector/
  data/
    validate.py              Dataset and YOLO-label validation
    preprocess.py            Image loading, label parsing and letterboxing
    split.py                 Reproducible, non-destructive split assignments
    augment.py               Image transforms with matching YOLO box updates
  train.py                   Configuration validation and model training
  inference.py               Image predictions and typed detections
  evaluate.py                Precision, recall and mAP evaluation
  visualize.py               Annotated images and evaluation metric plots
  export.py                  Local YOLO weights to ONNX export
tests/                       Synthetic-data and mocked unit tests
data/raw/                    Intended location for original data
data/processed/              Intended location for prepared data
outputs/                     Training runs, predictions and evaluation plots
notebooks/colab_train.ipynb  Colab workflow: download, validate, train, evaluate
```

## Setup

Use Python 3.10 or later, then install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
Ultralytics uses a CPU by default in `configs/train.yaml`; training, inference
and evaluation also accept an explicit `device="cpu"`.

## Dataset

The Colab workflow uses **MosquitoFusion**, a public dataset described as 1,204
images of mosquitoes, swarms, and breeding sites. The authors report an
87%/8%/5% train/validation/test split. Download it from
[Kaggle](https://www.kaggle.com/datasets/faiyazabdullah/mosquitofusion-dataset);
the [source repository](https://github.com/djairjr/MosquitoFusion) provides its
description and citation.

Kaggle lists the dataset license as MIT. The source repository refers to a
`LICENSE` file, but that file is not present in its current repository tree.
Review the Kaggle dataset terms and licensing details before use or
redistribution; the paper's Creative Commons license applies to the paper, not
automatically to the dataset. The dataset is downloaded at runtime and is not
committed here.

The checked-in `configs/data.yaml` remains a template. The Colab notebook
discovers the downloaded YOLO `data.yaml`, validates its paths and labels, then
uses a temporary configuration without changing the checked-in template.

Edit `configs/data.yaml` so its `path`, `train`, `val`, and optional `test`
entries point to existing image directories. Relative `path` values are
resolved from the configuration file's directory; split entries are resolved
from that root. `names` lists the class names by class ID. The checked-in YAML
is a template: its example split directories do not exist until a dataset is
prepared, so training and evaluation fail clearly until valid paths are set.

Dataset preparation helpers are available from
`src.mosquito_detector.data.preprocess` and
`src.mosquito_detector.data.split`. The splitter returns reproducible path
assignments with default train/validation/test ratios of 70/20/10; it does not
copy or move files.

## Data augmentation

Augmentations operate on a Pillow image and normalized `YoloAnnotation` values;
they return new image/annotation values and do not write over source files:

```python
from src.mosquito_detector.data.augment import (
    AugmentationConfig,
    augment_image,
)
from src.mosquito_detector.data.preprocess import YoloAnnotation, read_image

sample = augment_image(
    read_image("path/to/image.jpg"),
    [YoloAnnotation(0, 0.5, 0.5, 0.2, 0.3)],
    AugmentationConfig(
        horizontal_flip=True,
        rotation_degrees=5,
        brightness_factor=1.1,
        contrast_factor=0.9,
        scale_factor=1.05,
    ),
)
```

Rotation preserves the original canvas and clips transformed boxes to its
bounds. Scaling is centered: factors above one crop at the edges; factors below
one pad the original-sized canvas. Boxes with no visible area are omitted.
Choose augmentation ranges carefully; augmentation policy has not been tuned
against a real dataset.

## Reproduce training in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kadimane/mosquito-detection-yolo/blob/main/notebooks/colab_train.ipynb)

Open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) in Google
Colab, select an available GPU runtime, and run the cells in order. The Kaggle
access cell prompts for a Kaggle API token using hidden input; the token is
kept only in the current runtime's environment and must be entered again after
a runtime reset. Never put it in notebook code, Colab Secrets, Git, or chat.
The notebook downloads and validates the dataset, then trains pretrained
YOLOv8n for 30 epochs at 640px with batch size 8. It evaluates on the dataset's
held-out test split and generates annotated prediction examples, `metrics.png`,
and a `summary.json`. Download the results archive before the Colab runtime
ends.

The notebook uses this repository's dataset validation, training, inference,
and evaluation helpers. It does not modify the source dataset. To make the run
reviewable in GitHub, copy the generated `metrics.png`, `summary.json`, and
three sample prediction images into `results/` and update the project results
row below from `summary.json`. Do not present the paper's result as your own.

### Results

| Run | Evaluation split | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| This repository (YOLOv8n, 30 epochs) | Not run yet | — | — | — | — |
| MosquitoFusion paper (YOLOv8) | Paper-reported evaluation | 0.734 | 0.505 | 0.571 | Not reported in the abstract |

The paper reports its result in [arXiv:2404.01501](https://arxiv.org/abs/2404.01501).
It is a published reference, not an evaluation of this repository's model.

## Inference

Provide a local trained weights file and an image file or directory:

```python
from src.mosquito_detector.inference import run_inference

results = run_inference(
    weights="outputs/runs/mosquito-detection/weights/best.pt",
    source="path/to/images",
    output_dir="outputs/predictions",
    device="cpu",
)
for image_result in results:
    for detection in image_result.detections:
        print(detection.class_name, detection.confidence, detection.bbox)
```

Bounding boxes are pixel-space `(x1, y1, x2, y2)` coordinates. Annotated copies
are saved to the output directory; source images are left unchanged.

## Evaluation

Evaluation chooses the configured test split when available, otherwise the
validation split. Choose a split explicitly if needed:

```python
from src.mosquito_detector.evaluate import evaluate_model

result = evaluate_model(
    weights="outputs/runs/mosquito-detection/weights/best.pt",
    dataset_config_path="configs/data.yaml",
    output_dir="outputs/plots",
    split="test",
    device="cpu",
)
print(result.metrics.as_dict())
print(result.plot_path)
```

The reported precision, recall, mAP50 and mAP50-95 are read from the actual
Ultralytics evaluation result; a metric chart is saved with the evaluation
artifacts. This repository contains no evaluation results or model performance
claims.

## ONNX export

Export an existing local weights file; this operation does not train a model:

```python
from src.mosquito_detector.export import export_to_onnx

export = export_to_onnx(
    "outputs/runs/mosquito-detection/weights/best.pt",
    output_path="outputs/export/mosquito.onnx",
    image_size=640,
    device="cpu",
)
print(export.onnx_path)
```

The exporter requires local weights and the ONNX dependencies listed in
`requirements.txt` (`onnxslim` is used when simplification is enabled). It
validates input and output paths, does not overwrite an existing export unless
`overwrite=True`, and leaves the original weights unchanged. Export was not run
in this repository; no ONNX model or export performance is claimed. Options
such as dynamic dimensions, simplification and opset can be passed explicitly
to `export_to_onnx`.

## Tests

Run all unit tests with:

```bash
python -m unittest discover -s tests -v
```

Tests use temporary synthetic images, temporary YAML files and mocked YOLO
models. They do not download datasets or weights, train a model, or evaluate a
real dataset.

## Current limitations

- The MosquitoFusion dataset is not included; Kaggle currently lists its license
  as MIT, but its source repository does not contain the referenced `LICENSE`
  file. Review terms before redistribution.
- Training and real-data evaluation have not yet been run in this repository.
- Inference and evaluation require a locally available trained weights file.
- Augmentation functions are implemented and covered by synthetic tests, but
  their parameters have not been tuned on a selected dataset.
- ONNX export is implemented and mock-tested but has not been run on trained
  weights in this repository.
- Split generation returns assignments only; it does not materialize/copy a
  YOLO directory tree.
