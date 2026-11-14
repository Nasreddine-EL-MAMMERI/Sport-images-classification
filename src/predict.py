"""Génération d'un fichier de soumission.

Usage :
    python -m src.predict
    python -m src.predict --model models/best_model_ft100.keras \
                          --output submission_ft100.csv

Produit un CSV avec les colonnes ID et CLASS, dans l'ordre de test.csv.

Les deux options évitent d'écraser la soumission de référence : on peut
mesurer un modèle expérimental sans jamais toucher à `models/best_model.keras`
ni à `submission.csv`, c'est-à-dire sans jamais perdre la trace du run qui
a produit le score publié.

Une seule prédiction par lots de 32, au lieu de 1 115 appels à
model.predict() pour une image chacun.
"""

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import Config
from src.data import add_filepaths, load_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Config.BEST_MODEL_PATH,
        help="modèle à utiliser (défaut : models/best_model.keras)",
    )
    parser.add_argument(
        "--output", type=Path, default=Config.SUBMISSION_CSV,
        help="fichier de sortie (défaut : submission.csv)",
    )
    args = parser.parse_args()

    # ===== 1. Identifiants de test =====
    test_df = pd.read_csv(Config.TEST_CSV)
    expected_rows = len(test_df)
    test_ids = test_df["ID"].astype(str).tolist()

    # ===== 2. Chargement =====
    test_df = add_filepaths(test_df, Config.IMG_DIR)
    if test_df["filepath"].isna().any():
        missing = test_df.loc[test_df["filepath"].isna(), "ID"].tolist()[:5]
        raise FileNotFoundError(
            f"{int(test_df['filepath'].isna().sum())} images de test "
            f"introuvables. Exemples : {missing}"
        )

    X_test, _, _ = load_images(test_df, Config.IMG_SIZE, with_labels=False)

    # ===== 3. Modèle et classes =====
    if not args.model.exists():
        raise FileNotFoundError(f"Modèle introuvable : {args.model}")
    print(f"📦 Modèle : {args.model}")
    model = tf.keras.models.load_model(str(args.model))

    if not Config.CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"{Config.CLASS_NAMES_PATH} absent : lancez d'abord "
            "`python -m src.train`, qui enregistre l'ordre des classes."
        )
    with open(Config.CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    n_outputs = model.output_shape[-1]
    if n_outputs != len(class_names):
        raise ValueError(
            f"Le modèle prédit {n_outputs} classes, class_names.json en "
            f"contient {len(class_names)}."
        )

    # ===== 4. Prédiction par lots =====
    probs = model.predict(
        preprocess_input(X_test.astype(np.float32)),
        batch_size=Config.BATCH_SIZE,
        verbose=0,
    )
    predictions = [class_names[int(i)] for i in np.argmax(probs, axis=1)]

    # ===== 5. Contrôles avant écriture =====
    assert len(predictions) == expected_rows, (
        f"{len(predictions)} prédictions pour {expected_rows} lignes attendues"
    )
    assert set(predictions) <= set(class_names), "Label inconnu détecté"

    submission = pd.DataFrame({"ID": test_ids, "CLASS": predictions})
    submission.to_csv(args.output, index=False)

    print(f"✅ {args.output} — {len(submission)} lignes")
    print(submission["CLASS"].value_counts().head(5).to_string())

    # ===== 6. Comparaison au sample_submission, si disponible =====
    sample_path = Config.DATA_ROOT / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        same_columns = list(sample.columns) == list(submission.columns)
        same_ids = sample["ID"].astype(str).tolist() == test_ids
        print(f"\n🔎 Colonnes identiques au sample : {same_columns}")
        print(f"🔎 Même ordre d'ID : {same_ids}")
        if not (same_columns and same_ids):
            print("⚠️ La soumission risque d'être rejetée par Kaggle.")


if __name__ == "__main__":
    main()
