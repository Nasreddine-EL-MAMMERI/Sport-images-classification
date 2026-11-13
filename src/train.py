"""Entraînement de la phase 1 du notebook.

Usage (depuis la racine du projet) :
    python -m src.train

Produit :
    models/best_model.keras   meilleur modèle selon val_accuracy
    models/class_names.json   ordre des classes (indispensable à predict)
    models/history.json       courbes d'entraînement
"""

import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from config import Config
from src.data import load_datasets
from src.model import build_model, unfreeze_for_fine_tuning


def main() -> None:
    tf.keras.utils.set_random_seed(Config.SEED)

    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ===== 1. Données =====
    data = load_datasets(Config, load_test=False)
    class_names = data["class_names"]

    with open(Config.CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=1)

    # EfficientNetB0 normalise en interne ; preprocess_input ne remet PAS
    # les pixels dans [0, 1]. On l'appelle pour rester aligné sur le notebook.
    X_train = preprocess_input(data["X_train"].astype(np.float32))
    X_val = preprocess_input(data["X_val"].astype(np.float32))
    y_train, y_val = data["y_train"], data["y_val"]

    # ===== 2. Modèle =====
    model, base_model = build_model(data["n_classes"])
    model.summary()

    # ===== 3. Callbacks (valeurs du notebook) =====
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
        ),
        ModelCheckpoint(
            filepath=str(Config.BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=Config.REDUCE_LR_FACTOR,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=Config.MIN_LR,
            verbose=1,
        ),
    ]

    # ===== 4. Phase 1 : tête seule, backbone gelé =====
    print("\n=== PHASE 1 : entraînement de la tête ===")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS_PHASE1,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    history_all = dict(history.history)

    # ===== 5. Phase 2 : optionnelle, désactivée par défaut =====
    if Config.RUN_FINE_TUNING:
        print("\n=== PHASE 2 : fine-tuning du backbone ===")
        model = unfreeze_for_fine_tuning(model, base_model)

        history2 = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS_PHASE1 + Config.EPOCHS_PHASE2,
            initial_epoch=len(history_all["loss"]),
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        for key, values in history2.history.items():
            history_all[key] = history_all.get(key, []) + values

        model.save(str(Config.FINAL_MODEL_PATH))
    else:
        print(
            "\nℹ️ Phase 2 non exécutée : le score publié (0,92998) vient "
            "de la phase 1 seule. Mettre RUN_FINE_TUNING=True dans "
            "config.py pour l'activer — et re-mesurer le score."
        )

    with open(Config.HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history_all, f, indent=2)

    best = max(history_all["val_accuracy"])
    best_epoch = history_all["val_accuracy"].index(best) + 1
    print(f"\n✅ Meilleure exactitude de validation : {best:.4f} "
          f"(époque {best_epoch})")
    print(f"   Modèle : {Config.BEST_MODEL_PATH}")
    print(f"   Classes : {Config.CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
