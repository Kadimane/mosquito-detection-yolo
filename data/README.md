# Dataset

The reproducible Colab workflow uses [MosquitoFusion on Kaggle](https://www.kaggle.com/datasets/faiyazabdullah/mosquitofusion-dataset),
described by its authors as 1,204 images with mosquito, swarm, and breeding-site
classes. The Kaggle page lists the dataset license as MIT. The authors'
[source repository](https://github.com/djairjr/MosquitoFusion) references a
`LICENSE` file that is absent from its current file listing, so review the
dataset page and terms before use or redistribution. The accompanying paper is
available at <https://arxiv.org/abs/2404.01501>; its paper license does not
automatically cover the dataset.

The dataset is downloaded at runtime and is not stored in this repository.
Keep local source files in `data/raw/` and prepared copies in `data/processed/`;
dataset contents are excluded from Git by `.gitignore`. The Colab notebook
validates the downloaded dataset and creates a temporary YOLO configuration;
the checked-in `configs/data.yaml` remains a template.

The authors report an 87%/8%/5% train/validation/test split. Their paper reports
YOLOv8 precision 0.734, recall 0.505, and mAP@50 0.571. These are reference
results only; this project has not yet trained or evaluated its own model.

Citation:

```bibtex
@inproceedings{sayeedi2024mosquitofusion,
  title={MosquitoFusion: A Multiclass Dataset for Real-Time Detection of Mosquitoes, Swarms, and Breeding Sites Using Deep Learning},
  author={Md. Faiyaz Abdullah Sayeedi and Fahim Hafiz and Md Ashiqur Rahman},
  booktitle={The Second Tiny Papers Track at ICLR 2024},
  year={2024},
  url={https://openreview.net/forum?id=3s4hFx8pYs}
}
```
