import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.data import get_datasets
from config import Config


def main():
    # Charger les données de validation
    _, val_ds = get_datasets(
        str(Config.TRAIN_DIR),
        validation_split=Config.VALIDATION_SPLIT
    )

    class_names = val_ds.class_names
    model = tf.keras.models.load_model("models/best_model.keras")

    # Prédictions
    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)
    y_pred = model.predict(val_ds)
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Rapport texte
    print(classification_report(y_true_labels, y_pred_labels,
                                target_names=class_names))

    # Matrice de confusion
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.title("Matrice de Confusion")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    acc = np.mean(y_true_labels == y_pred_labels)
    print(f"✅ Accuracy sur validation : {acc:.4f}")


if __name__ == "__main__":
    main()