"""Configuration du projet — reprise exacte du notebook Kaggle.

Le notebook qui a produit 0,92998 est la référence. Toutes les valeurs
ci-dessous en sont extraites telles quelles.
"""

import os
from dataclasses import dataclass
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle/input")

COMPETITION_SLUG = "opencv-tf-project-2-image-classification-round-4"


def resolve_data_root() -> Path:
    """Trouve le dossier de la compétition, quel que soit le montage Kaggle.

    Kaggle a changé son arborescence : les compétitions sont désormais
    montées dans /kaggle/input/competitions/<slug> et non plus dans
    /kaggle/input/<slug>. Coder le chemin en dur casse le projet au
    moindre changement de plateforme.

    On teste donc les emplacements connus, puis, en dernier recours, on
    cherche un train.csv dont le dossier parent ressemble à la compétition.
    """
    candidates = [
        Path("/kaggle/input/competitions") / COMPETITION_SLUG,  # montage actuel
        Path("/kaggle/input") / COMPETITION_SLUG,                # ancien montage
        Path("data"),                                            # hors Kaggle
    ]

    for candidate in candidates:
        if (candidate / "train.csv").exists():
            print(f"📁 DATA_ROOT = {candidate}")
            return candidate

    # Dernier recours (Kaggle seulement) : chercher le CSV plutôt que de
    # deviner le chemin. Hors Kaggle on retombe sur le dossier local `data`.
    if IS_KAGGLE:
        matches = sorted(Path("/kaggle/input").glob("*/train.csv"))
        matches += sorted(Path("/kaggle/input").glob("*/*/train.csv"))
        for match in matches:
            if "opencv" in str(match.parent).lower() or list(
                match.parent.glob("sports_images")
            ):
                print(f"📁 DATA_ROOT = {match.parent} (trouvé par recherche)")
                return match.parent
        if matches:
            print(f"📁 DATA_ROOT = {matches[0].parent} (trouvé par recherche)")
            return matches[0].parent

    return Path("data")


@dataclass
class Config:
    # ===== CHEMINS =====
    DATA_ROOT: Path = resolve_data_root()

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
    # Laisser à False pour reproduire le résultat annoncé.
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
