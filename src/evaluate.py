"""Évaluation sur le jeu de validation.

Usage :
    python -m src.evaluate
    python -m src.evaluate --model models/best_model_ft100.keras \
                           --output results/confusion_matrix_ft100.png

Produit dans results/ :
    confusion_matrix.png        matrice 73×73 réordonnée par regroupement
    classification_report.txt   précision / rappel / F1 par classe
    top_errors.txt              erreurs par classe et paires les plus confondues

POURQUOI CETTE VERSION CHANGE TOUT

1. ORDONNANCEMENT PAR REGROUPEMENT HIÉRARCHIQUE.
   Une matrice triée alphabétiquement disperse les classes qui se confondent :
   `luge` et `bobsled` se retrouvent à quarante lignes l'un de l'autre, et la
   structure de l'erreur est invisible. On calcule ici une distance entre
   classes à partir des confusions mutuelles, puis un linkage average avec
   ordonnancement des feuilles. Les blocs de confusion apparaissent.

2. ÉCHELLE DE COULEUR TRONQUÉE À 0,30, et c'est assumé.
   La diagonale vaut 1,00 partout : avec une échelle 0–1, elle écrase tout le
   reste et les erreurs, qui tournent autour de 0,02, deviennent invisibles.
   On tronque donc, et on l'écrit dans la légende.

3. DEUX DIAGNOSTICS À CÔTÉ.
   Le taux d'erreur par classe ne dit pas *vers quoi* le modèle se trompe.
   Les paires (réelle → prédite) le disent, et ce sont elles qui suggèrent une
   action : fusionner, rééquilibrer, ou accepter.

4. PRÉDICTION SUR LES ENTRÉES PRÉTRAITÉES.
   Le notebook d'origine calculait sa matrice sur `X_val` brut, sans
   `preprocess_input`. Ses chiffres par classe ne sont donc pas fiables.
   Ici tout passe par le même pipeline que l'entraînement.

La version précédente affichait 73×73 = 5 329 nombres annotés dans une figure
de 8×6 pouces : illisible.
"""

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import Config
from src.data import load_datasets

CMAP = "Blues"
VMAX = 0.30          # échelle tronquée : sans ça, la diagonale écrase les erreurs
N_TOP = 15


# =====================================================================
# Ordonnancement des classes
# =====================================================================
def cluster_order(cm_norm: np.ndarray) -> np.ndarray:
    """Réordonne les classes pour que celles qui se confondent soient voisines.

    Distance entre deux classes i et j : 1 − moyenne des confusions mutuelles
    (i prédit j, j prédit i). Deux classes qui ne se confondent jamais sont à
    distance 1 ; deux classes indiscernables sont à distance 0.
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    sym = (cm_norm + cm_norm.T) / 2.0
    dist = np.clip(1.0 - sym, 0.0, None)
    np.fill_diagonal(dist, 0.0)          # squareform exige une diagonale nulle

    tree = linkage(squareform(dist, checks=False), method="average")
    return leaves_list(tree)


def top_pairs(cm_counts: np.ndarray, cm_norm: np.ndarray,
              class_names: list, n: int = N_TOP) -> list:
    """Les n paires (réelle → prédite) les plus fréquentes, hors diagonale."""
    rows = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_counts[i, j] > 0:
                rows.append((
                    class_names[i], class_names[j],
                    int(cm_counts[i, j]), float(cm_norm[i, j]), i, j,
                ))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:n]


# =====================================================================
# Graphique
# =====================================================================
def plot_diagnostics(cm_counts: np.ndarray, cm_norm: np.ndarray,
                     class_names: list, out_path: Path, title: str) -> list:
    order = cluster_order(cm_norm)
    cm_sorted = cm_norm[np.ix_(order, order)]
    names_sorted = [class_names[i] for i in order]

    n = len(class_names)
    error_rate = 1.0 - np.diag(cm_norm)
    err_order = np.argsort(error_rate)[::-1]
    pairs = top_pairs(cm_counts, cm_norm, class_names)

    fig = plt.figure(figsize=(18, 9.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0],
                          height_ratios=[1, 1], wspace=0.30, hspace=0.45)
    ax_m = fig.add_subplot(gs[:, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_pair = fig.add_subplot(gs[1, 1])

    # ---------- A. Matrice réordonnée ----------
    im = ax_m.imshow(cm_sorted, cmap=CMAP, vmin=0, vmax=VMAX,
                     interpolation="nearest", aspect="equal")
    ax_m.set_xticks([])
    ax_m.set_yticks(range(n))
    ax_m.set_yticklabels(names_sorted, fontsize=5)
    ax_m.tick_params(axis="y", length=0, pad=1)
    ax_m.set_xlabel(f"Predicted class — {n} classes", fontsize=9)
    ax_m.set_ylabel("True class", fontsize=9)
    ax_m.set_title(title, fontsize=11, loc="left")

    cbar = fig.colorbar(im, ax=ax_m, fraction=0.030, pad=0.015)
    cbar.set_label("share of the true class", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ---------- B. Taux d'erreur par classe ----------
    top = err_order[:N_TOP]
    y = np.arange(N_TOP)[::-1]
    ax_err.barh(y, error_rate[top] * 100, color="#c0392b", alpha=0.85,
                height=0.7)
    ax_err.set_yticks(y)
    ax_err.set_yticklabels([class_names[i] for i in top], fontsize=7.5)
    ax_err.set_xlabel("error rate (%)", fontsize=8)
    ax_err.set_title(f"Top {N_TOP} classes by error rate", fontsize=10,
                     loc="left")
    ax_err.grid(axis="x", alpha=0.25)
    ax_err.set_xlim(0, max(error_rate[top] * 100) * 1.22)
    for yi, idx in zip(y, top):
        ax_err.text(error_rate[idx] * 100 + 0.35, yi,
                    f"{error_rate[idx] * 100:.1f}", va="center", fontsize=7)

    # ---------- C. Paires les plus confondues ----------
    y = np.arange(len(pairs))[::-1]
    values = [p[2] for p in pairs]
    ax_pair.barh(y, values, color="#8e44ad", alpha=0.85, height=0.7)
    ax_pair.set_yticks(y)
    ax_pair.set_yticklabels([f"{p[0]} → {p[1]}" for p in pairs], fontsize=7)
    ax_pair.set_xlabel("number of images", fontsize=8)
    ax_pair.set_title(f"Top {len(pairs)} confusions (true → predicted)",
                      fontsize=10, loc="left")
    ax_pair.grid(axis="x", alpha=0.25)
    if values:
        ax_pair.set_xlim(0, max(values) * 1.16)
    for yi, p in zip(y, pairs):
        ax_pair.text(p[2] + max(values) * 0.015, yi,
                     f"{p[2]}  ({p[3] * 100:.1f} %)", va="center", fontsize=7)

    fig.suptitle("Where the model fails", fontsize=13, fontweight="bold",
                 y=0.985)
    fig.text(
        0.012, -0.012,
        f"Rows and columns are ordered by hierarchical clustering of mutual "
        f"confusions, not alphabetically. Colour scale clipped at {VMAX:.2f} "
        f"so that off-diagonal errors stay visible; the diagonal (1.00) "
        f"saturates.",
        fontsize=7.5, style="italic", color="#555",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {out_path}")
    return pairs


# =====================================================================
# Principal
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Config.BEST_MODEL_PATH)
    parser.add_argument("--output", type=Path,
                        default=Config.RESULTS_DIR / "confusion_matrix.png")
    args = parser.parse_args()

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_datasets(Config, load_test=False)
    class_names = data["class_names"]

    if not args.model.exists():
        raise FileNotFoundError(f"Modèle introuvable : {args.model}")
    print(f"📦 Modèle : {args.model}")
    model = tf.keras.models.load_model(str(args.model))

    X_val = preprocess_input(data["X_val"].astype(np.float32))
    y_true = data["y_val"].argmax(axis=1)

    probs = model.predict(X_val, verbose=0, batch_size=Config.BATCH_SIZE)
    y_pred = probs.argmax(axis=1)

    accuracy = float((y_true == y_pred).mean())
    print(f"\n✅ Exactitude de validation : {accuracy:.4f} "
          f"({accuracy * 100:.2f} %)")

    # ---------- Rapport par classe ----------
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    print(report)
    (Config.RESULTS_DIR / "classification_report.txt").write_text(
        f"Modèle : {args.model}\nExactitude de validation : {accuracy:.4f}\n\n"
        f"{report}",
        encoding="utf-8",
    )

    # ---------- Matrices ----------
    cm_counts = confusion_matrix(y_true, y_pred)
    cm_norm = cm_counts.astype("float") / cm_counts.sum(axis=1, keepdims=True)

    pairs = plot_diagnostics(
        cm_counts, cm_norm, class_names, args.output,
        title=f"{args.model.stem} · {len(y_true)} validation images · "
              f"accuracy {accuracy:.4f}",
    )

    # ---------- Diagnostics texte ----------
    error_rate = 1.0 - np.diag(cm_norm)
    order = np.argsort(error_rate)[::-1]

    lines = [f"Modèle : {args.model}",
             f"Exactitude de validation : {accuracy:.4f}", "",
             "Taux d'erreur par classe (du plus mauvais au meilleur)", ""]
    for idx in order:
        lines.append(
            f"{class_names[idx]:25s} erreur {error_rate[idx] * 100:5.1f} %  "
            f"(support {int(cm_counts[idx].sum())})"
        )

    lines += ["", "Paires les plus confondues (réelle → prédite)", ""]
    for real, pred, n_img, frac, _, _ in pairs:
        lines.append(
            f"{real:25s} → {pred:25s} {n_img:3d} images "
            f"({frac * 100:.1f} % de la classe)"
        )

    (Config.RESULTS_DIR / "top_errors.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("🔍 10 classes les plus confondues :")
    for idx in order[:10]:
        print(f"   {class_names[idx]:25s} {error_rate[idx] * 100:5.1f} %")

    print("\n🔁 10 principales confusions :")
    for real, pred, n_img, frac, _, _ in pairs[:10]:
        print(f"   {real} → {pred} : {n_img} images ({frac * 100:.1f} %)")


if __name__ == "__main__":
    main()
