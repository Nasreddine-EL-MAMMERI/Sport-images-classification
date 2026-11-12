# 🏆 Image Classification — Sport Images (Kaggle GPU)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF)
![Public LB](https://img.shields.io/badge/Public%20Leaderboard-0.92998-green)

> Fine-tuning d'EfficientNetB0 pour la classification d'images de sport.
> **0.92998 Public Leaderboard — 93,0%** 🏆

## 📊 Dataset
Source : [OpenCV TF Project 2 — Image Classification Round 4](https://www.kaggle.com/competitions/opencv-tf-project-2-image-classification-round-4)

- 73 classes de sport 
- 10 031 images d'entraînement + 1 115 images de test
- Split 80% train / 20% validation

> ⚠️ Le dataset n'est pas inclus dans ce dépôt (trop lourd pour Git).

## 🧠 Approche
- Backbone **EfficientNetB0** pré-entraîné sur ImageNet
- tête de classification uniquement (backbone gelé) → 95,67% val
- Prétraitement `preprocess_input` (EfficientNet)
- Data augmentation + EarlyStopping + ReduceLROnPlateau

## 🚀 Utilisation
```bash
git clone https://github.com/Nasreddine-EL-MAMMERI/Sport-images-classification.git
cd Sport-images-classification
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt