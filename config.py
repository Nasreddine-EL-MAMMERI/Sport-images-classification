import os
from dataclasses import dataclass
from pathlib import Path

# Détecte si on est sur Kaggle ou en local
IS_KAGGLE = os.path.exists("/kaggle/input")


@dataclass
class Config:
    # ===== CHEMINS =====
    if IS_KAGGLE:
        # ⚠️ À VÉRIFIER sur ta page Kaggle Data : le dossier peut s'appeler "dataset" ou être à la racine
        DATA_ROOT = Path("/kaggle/input/opencv-tf-project-2-image-classification-round-4")
        # Si les dossiers Train/ et Test/ sont directement à la racine du dataset :
        TRAIN_DIR = DATA_ROOT / "Train"
        TEST_DIR = DATA_ROOT / "Test"
        TEST_CSV = DATA_ROOT / "test.csv"
    else:
        # En local (GitHub) : pas de data, juste la structure
        DATA_ROOT = Path("data")
        TRAIN_DIR = DATA_ROOT / "Train"
        TEST_DIR = DATA_ROOT / "Test"
        TEST_CSV = DATA_ROOT / "test.csv"

    SUBMISSION_CSV = Path("submission.csv")

    # ===== MODÈLE =====
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    SEED = 42
    NUM_CLASSES = None  # Détecté automatiquement depuis les dossiers

    # ===== ENTRAÎNEMENT =====
    EPOCHS_PHASE1 = 10      # Uniquement la tête de classification
    EPOCHS_PHASE2 = 10      # Fine-tuning du backbone
    LR_PHASE1 = 1e-3
    LR_PHASE2 = 1e-5
    FINE_TUNE_AT = 100      # Débloque les couches à partir de cet indice

    # ===== VALIDATION =====
    VALIDATION_SPLIT = 0.2  # 20% du dossier Train servent de validation

    # ===== AUGMENTATION =====
    USE_AUGMENTATION = True