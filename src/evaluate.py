"""Évaluation sur le jeu de validation.

Usage :
    python -m src.evaluate

Produit dans results/ :
    classification_report.txt  précision / rappel / F1 par classe
    confusion_matrix.png       matrice 73×73 NORMALISÉE, sans annotations
    top_errors.txt             les 20 classes les plus confondues

"""

import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import Config
from src.data import load_datasets


def main() -> None:
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_datasets(Config, load_test=False)
    class_names = data["class_names"]

    model = tf.keras.models.load_model(str(Config.BEST_MODEL_PATH))

    X_val = preprocess_input(data["X_val"].astype(np.float32))
    y_true = data["y_val"].argmax(axis=1)

    probs = model.predict(X_val, verbose=0)
    y_pred = probs.argmax(axis=1)

    accuracy = float((y_true == y_pred).mean())
    print(f"\n✅ Exactitude de validation : {accuracy:.4f} ({accuracy*100:.2f}%)")

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    print(report)
    (Config.RESULTS_DIR / "classification_report.txt").write_text(
        f"Exactitude de validation : {accuracy:.4f}\n\n{report}", encoding="utf-8"
    )

    # ===== Matrice normalisée, sans annotations =====
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion de la classe réelle"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title(
        f"Matrice de confusion — validation ({len(y_true)} images, "
        f"normalisée par classe)"
    )
    ax.set_xlabel(f"Classe prédite (73 classes, ordre alphabétique)")
    ax.set_ylabel(f"Classe réelle (73 classes, ordre alphabétique)")
    fig.tight_layout()
    fig.savefig(Config.RESULTS_DIR / "confusion_matrix.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"✅ results/confusion_matrix.png")

    # ===== Diagnostic : classes les plus confondues =====
    error_rate = 1.0 - np.diag(cm_norm)
    order = np.argsort(error_rate)[::-1]

    lines = ["Taux d'erreur par classe (du plus mauvais au meilleur)", ""]
    for idx in order:
        lines.append(
            f"{class_names[idx]:25s} erreur {error_rate[idx]*100:5.1f}%  "
            f"(support {cm[idx].sum()})"
        )

    (Config.RESULTS_DIR / "top_errors.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\n🔍 15 classes les plus confondues :")
    for idx in order[:15]:
        print(f"   {class_names[idx]:25s} {error_rate[idx]*100:5.1f}%")

    # ===== Paires les plus confondues =====
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    pairs = []
    for i, j in zip(*np.unravel_index(np.argsort(off_diag, axis=None)[::-1],
                                      off_diag.shape)):
        if off_diag[i, j] == 0 or len(pairs) >= 10:
            break
        pairs.append((class_names[i], class_names[j], int(off_diag[i, j])))

    if pairs:
        print("\n🔁 Principales confusions :")
        for real, pred, n in pairs:
            print(f"   {real} → {pred} : {n}")


if __name__ == "__main__":
    main()
