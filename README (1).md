# Ovarian Cancer Detection using ResNet50

Deep learning project that classifies ovarian histopathology images with transfer learning on **ResNet50** (pretrained on ImageNet).

> **Disclaimer:** This project is for research and educational purposes only. It is not a medical device and must not be used for diagnosis.

## Approach

1. Load images from class folders and apply augmentation (flips, rotation, zoom, contrast).
2. Use ResNet50 as a frozen feature extractor and train a new classification head.
3. Fine-tune the top 30 ResNet50 layers at a low learning rate (BatchNorm layers stay frozen).
4. Handle class imbalance with class weights.
5. Evaluate with precision, recall, F1, ROC-AUC and a confusion matrix.

## Dataset

Use any ovarian histopathology image dataset arranged in class folders, for example:

- [UBC Ovarian Cancer Subtype Classification (Kaggle)](https://www.kaggle.com/competitions/UBC-OCEAN)
- Ovarian cancer histopathology datasets on Mendeley Data or Kaggle

Layout (folder names become the class labels):

```
data/
  train/  cancer/  non_cancer/
  val/    cancer/  non_cancer/
  test/   cancer/  non_cancer/     # optional
```

or a single folder with only class subfolders, which is split 70/15/15 automatically.

## Setup and run

```bash
pip install -r requirements.txt
python ovarian_cancer_resnet50.py --data data
```

Options: `--epochs`, `--fine-tune-epochs`, `--img-size`, `--batch-size`, `--unfreeze-layers`.

Outputs are saved to `outputs/`: training curves, confusion matrices, `best_model.keras`, and `class_names.json`.

## Predict on one image

```bash
python predict.py path/to/image.jpg
```

## Project structure

```
.
├── data/
├── outputs/
├── ovarian_cancer_resnet50.py
├── predict.py
├── requirements.txt
└── README.md
```

## Possible improvements

- Add Grad-CAM to visualize which regions drive predictions
- Patient-level splits to avoid data leakage between train and test
- Compare with EfficientNet or ViT
- Stain normalization for histology images
