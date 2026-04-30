# BT5153 Group Project — Toxic Comment Detection

## Overview

This project builds a multilingual toxic comment detection system. Given an input comment, the model outputs a toxicity probability and a binary prediction based on an optimised threshold.

## Repository Structure

```
├── 01_preprocessing_EDA.ipynb   # Data loading, preparation, and exploratory analysis
├── 02_modeling.ipynb            # Model training (TF-IDF+LR, BiLSTM, mBERT, XLM-R)
├── 03_evaluation.ipynb          # Threshold optimisation, final evaluation, and test set performance
└── README.md
```

## Data

Data is from the [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data) Kaggle competition.

Due to file size constraints, `data05.zip` contains:
- `prepared data/` — preprocessed training/dev/validation splits (output of Module 1–2)
- `test.csv` — raw test set
- `test_labels.csv` — test set labels

The original Kaggle raw files (`jigsaw-toxic-comment-train.csv`, 
`jigsaw-unintended-bias-train.csv`) are not included.
To reproduce Module 1 from scratch, download them from:
https://www.kaggle.com/competitions/jigsaw-multilingual-toxic-comment-classification/data

## Environment

All notebooks are designed to run on **Google Colab**.

Install dependencies:
```python
pip install transformers datasets scikit-learn torch tensorflow
```

## How to Run

Run the notebooks **in order**:

1. **`01_preprocessing_EDA.ipynb`** — loads raw data, cleans and prepares it, saves processed files to Google Drive
2. **`02_modeling.ipynb`** — trains all models, saves prediction files to Google Drive
3. **`03_evaluation.ipynb`** — loads prediction files from step 2, runs threshold optimisation, final multilingual evaluation, and test set performance

## Path Configuration

Before running, update the `DATA_PATH` / `PROJECT_ROOT` variables in each notebook to match your own Google Drive folder path:

```python
# In 01_preprocessing_EDA.ipynb
TOXIC_TRAIN_PATH = "/content/drive/My Drive/<your-folder>/jigsaw-toxic-comment-train.csv"

# In 02_modeling.ipynb and 03_evaluation.ipynb
PROJECT_ROOT = "/content/drive/My Drive/<your-folder>"
```

## Models

| Model | Type |
|-------|------|
| TF-IDF + Logistic Regression | Baseline (supervised) |
| BiLSTM | Deep learning (supervised) |
| mBERT | Transformer (multilingual) |
| XLM-R | Transformer (multilingual, stronger baseline) |
