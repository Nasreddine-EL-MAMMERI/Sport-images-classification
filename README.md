# Image Classification — Fine-Tuning with TensorFlow / Keras

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF)

> Fine-tuning d'un modèle pré-entraîné (EfficientNetB0) pour la classification d'images.
> Développé localement, entraîné sur **Kaggle GPU**, soumis sur **Kaggle**.

## 📊 Dataset
Source : [Kaggle Competition — OpenCV TF Project 2](https://www.kaggle.com/competitions/opencv-tf-project-2-image-classification-round-4)

- Dossier `Train/` : images d'entraînement (split automatique 80/20 train/validation)
- Dossier `Test/` : images de test pour la soumission finale

> ⚠️ Le dataset n'est pas inclus dans ce dépôt (trop lourd pour Git). Téléchargez-le directement sur Kaggle.

## 🧠 Approche
1. **Phase 1** : Entraînement de la tête de classification uniquement (backbone gelé)
2. **Phase 2** : Fine-tuning des dernières couches du backbone EfficientNetB0
3. **Augmentation de données** intégrée (flip, rotation, zoom, contraste)
4. **Soumission** : génération de `submission.csv` pour Kaggle

## 🚀 Utilisation locale (structure uniquement)
```bash
git clone https://github.com/Nasreddine-EL-MAMMERI/Sport-images-classification.git
cd Image-Classifier
python -m venv .venv
.\.venv\Scripts\activate      # Windows
pip install -r requirements.txt
