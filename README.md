# Détection de pneumonie sur radiographies thoraciques

Projet Deep Learning end-to-end : classification binaire (Normal vs Pneumonie)
sur le dataset *Chest X-Ray Images (Pneumonia)*.

## Installation
```bash
uv sync --extra explain
```

## Données
```bash
cp .env.example .env   # puis renseigne tes identifiants Kaggle
python download_data.py
```

## Pipeline
1. EDA & re-split stratifié
2. Baseline CNN from scratch
3. Transfer learning (DenseNet121 / ResNet50 / EfficientNet)
4. Évaluation (ROC, PR, matrice de confusion, recall-first)
5. Interprétabilité (Grad-CAM)
