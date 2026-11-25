# Mosquito Detection with YOLO

This repository contains reusable dataset utilities and an Ultralytics YOLO
workflow for object detection. No dataset or trained weights are included.

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
  train.py                   Configuration validation and model training
  inference.py               Image predictions and typed detections
  evaluate.py                Precision, recall and mAP evaluation
  visualize.py               Annotated images and evaluation metric plots
tests/                       Synthetic-data and mocked unit tests
data/raw/                    Intended location for original data
data/processed/              Intended location for prepared data
outputs/                     Training runs, predictions and evaluation plots
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

## Dataset configuration

No dataset has been downloaded or selected. Obtain and review a dataset license
before use. Keep original files in `data/raw/` and prepare copies under
`data/processed/`; these utilities do not download data or alter source images
and labels.

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

## Training

Review `configs/train.yaml` and `configs/data.yaml`, then call:

```python
from src.mosquito_detector.train import train_model

result = train_model()
print(result.output_dir)
```

Training configuration controls weights/model definition, epochs, image size,
batch size, device, output directory and run name. Ultralytics writes training
artifacts and checkpoints under the configured output location. Training was
not run as part of this repository setup.

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

## Tests

Run all unit tests with:

```bash
python -m unittest discover -s tests -v
```

Tests use temporary synthetic images, temporary YAML files and mocked YOLO
models. They do not download datasets or weights, train a model, or evaluate a
real dataset.

## Current limitations

- Dataset license, source, annotations and train/validation/test split remain
  unselected and unverified.
- Training and real-data evaluation have not been run.
- Inference and evaluation require a locally available trained weights file.
- Split generation returns assignments only; it does not materialize/copy a
  YOLO directory tree.
