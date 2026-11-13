import os
from dataclasses import dataclass
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle/input")


@dataclass
class Config:
    # ===== CHEMINS =====
    if IS_KAGGLE:
        DATA_ROOT: Path = Path(
            "/kaggle/input/opencv-tf-project-2-image-classification-round-4"
        )
    else:
        DATA_ROOT: Path = Path("data")

    TRAIN_CSV: Path = DATA_ROOT / "train.csv"
    TEST_CSV: Path = DATA_ROOT / "test.csv"
    IMG_DIR: Path = DATA_ROOT / "sports_images" / "all_images"

    MODELS_DIR: Path = Path("models")
    RESULTS_DIR: Path = Path("results")
    SUBMISSION_CSV: Path = Path("submission.csv")

    BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.keras"
    FINAL_MODEL_PATH: Path = MODELS_DIR / "final_model.keras"
    CLASS_NAMES_PATH: Path = MODELS_DIR / "class_names.json"
    HISTORY_PATH: Path = MODELS_DIR / "history.json"

    # ===== DONNÉES =====
    IMG_SIZE: tuple = (224, 224)
    BATCH_SIZE: int = 32
    SEED: int = 42
    VALIDATION_SPLIT: float = 0.2

    # ===== ENTRAÎNEMENT =====
    EPOCHS_PHASE1: int = 20
    LEARNING_RATE_PHASE1: float = 1e-3

    # Phase 2 (fine-tuning du backbone) : présente dans le notebook sous
    # forme de fonction, mais JAMAIS APPELÉE. Le score publié vient donc
    # de la phase 1 seule — extraction de features, backbone gelé.
   
    RUN_FINE_TUNING: bool = False
    EPOCHS_PHASE2: int = 20
    LEARNING_RATE_PHASE2: float = 1e-5
    FINE_TUNE_AT: int = 100  # nombre de couches dégelées en partant de la fin

    # ===== CALLBACKS =====
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    REDUCE_LR_PATIENCE: int = 5
    REDUCE_LR_FACTOR: float = 0.3
    MIN_LR: float = 1e-7
