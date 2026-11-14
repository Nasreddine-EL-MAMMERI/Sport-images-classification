"""Courbes d'entraînement de la phase 2, lues depuis les vrais fichiers.

Usage (depuis la racine du projet, après `python -m src.finetune`) :
    python -m src.plot_curves

Lit `models/history_ft*.json` — les fichiers écrits par src.finetune — et
produit `results/finetuning_curves.png`.

Contrairement à `src/plot_summary.py`, ce script ne contient AUCUNE valeur
en dur : relancer un entraînement puis ce script suffit à mettre la figure à
jour. C'est la version à utiliser ; les valeurs transcrites à la main dans
plot_summary.py ne servent qu'aux panneaux qui dépendent du classement Kaggle.

CE QUE LA FIGURE MONTRE, ET LE PIÈGE QU'ELLE DOCUMENTE
    Deux couches Dropout(0.5) sont actives pendant l'entraînement et
    désactivées à la validation. L'exactitude d'entraînement est donc mesurée
    sur un réseau partiellement éteint : elle est structurellement PLUS BASSE
    que celle de validation. L'écart train/val n'est pas un indicateur de
    sur-apprentissage dans cette architecture.
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

from config import Config

TRAIN_COLOR = "#d98841"
VAL_COLOR = "#1f4e79"
BEST_COLOR = "#27ae60"


def load_histories() -> dict:
    """Charge tous les models/history_ft*.json et ne garde que les listes."""
    runs = {}
    paths = sorted(Config.MODELS_DIR.glob("history_ft*.json"))

    if not paths:
        raise FileNotFoundError(
            f"Aucun historique dans {Config.MODELS_DIR}/. "
            "Lancez d'abord `python -m src.finetune --unfreeze 100`."
        )

    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        series = {k: v for k, v in raw.items() if isinstance(v, list)}
        if "accuracy" not in series or "val_accuracy" not in series:
            print(f"⚠️  {path.name} : accuracy/val_accuracy absents — ignoré")
            continue

        tag = path.stem.replace("history_", "")
        series["_meta"] = {
            k: raw[k] for k in ("unfreeze", "learning_rate_phase2",
                                "duration_seconds")
            if k in raw
        }
        runs[tag] = series
        print(f"📈 {path.name} — {len(series['accuracy'])} époques")

    if not runs:
        raise ValueError("Aucun historique exploitable.")

    # Ordre naturel : ft30 avant ft100, pas l'ordre alphabétique.
    return dict(sorted(runs.items(),
                       key=lambda kv: kv[1]["_meta"].get("unfreeze", 10**6)))


def print_table(runs: dict) -> None:
    """Résumé chiffré, et l'écart train−val époque par époque."""
    print("\n" + "=" * 72)
    print(f"{'run':8s} {'époques':>8s} {'best val':>9s} {'époque':>7s} "
          f"{'train fin':>10s} {'écart fin':>10s}")
    print("-" * 72)

    for tag, s in runs.items():
        train = np.array(s["accuracy"])
        val = np.array(s["val_accuracy"])
        best_epoch = int(np.argmax(val)) + 1

        print(f"{tag:8s} {len(train):>8d} {val[best_epoch - 1]:>9.4f} "
              f"{best_epoch:>7d} {train[-1]:>10.4f} "
              f"{100 * (train[-1] - val[-1]):>+9.2f} pt")

        print(f"   écart train−val par époque : "
              + " ".join(f"{100 * (t - v):+.1f}" for t, v in zip(train, val)))
    print("=" * 72)
    print("Rappel : écart négatif = validation au-dessus de l'entraînement.")
    print("C'est le Dropout(0.5), actif à l'entraînement et désactivé en "
          "validation.\nCe n'est PAS un signe de bonne santé du modèle.\n")


def plot(runs: dict) -> None:
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 4.8), sharey=True,
                             squeeze=False)
    axes = axes[0]

    for ax, (tag, s) in zip(axes, runs.items()):
        train = np.array(s["accuracy"])
        val = np.array(s["val_accuracy"])
        epochs = np.arange(1, len(train) + 1)

        ax.plot(epochs, train, "--o", color=TRAIN_COLOR, markersize=4,
                linewidth=1.6, label="training accuracy (Dropout ON)")
        ax.plot(epochs, val, "-o", color=VAL_COLOR, markersize=4,
                linewidth=2.0, label="validation accuracy (Dropout OFF)")

        best = int(np.argmax(val))
        ax.plot(epochs[best], val[best], "o", color=BEST_COLOR, markersize=11,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        offset = 3.6
        x_text = epochs[best] - offset
        ha = "right"
        if x_text < 1.5:                      # trop près du bord gauche
            x_text = epochs[best] + 1.2
            ha = "left"
        ax.annotate(
            f"best val · epoch {epochs[best]}\n{val[best]:.4f}",
            xy=(epochs[best], val[best]),
            xytext=(x_text, val[best] + 0.004),
            ha=ha, va="center", fontsize=8.5, color="#1e7a45",
            arrowprops=dict(arrowstyle="->", color=BEST_COLOR, lw=1.2,
                            shrinkA=2, shrinkB=6),
        )

        depth = s["_meta"].get("unfreeze")
        title = f"{tag}" + (f" — last {depth} layers unfrozen" if depth else "")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs[:: max(1, len(epochs) // 8)])
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Accuracy")
    lo = min(min(s["accuracy"]) for s in runs.values())
    hi = max(max(s["val_accuracy"]) for s in runs.values())
    axes[0].set_ylim(lo - 0.005, hi + 0.008)
    axes[0].legend(fontsize=8.5, loc="lower right", framealpha=0.95)

    fig.suptitle(
        "Validation sits above training for most of the run — that is "
        "Dropout(0.5), not a healthy model",
        fontsize=11.5, y=1.02,
    )

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()

    out = Config.RESULTS_DIR / "finetuning_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {out}")


def main() -> None:
    runs = load_histories()
    print_table(runs)
    plot(runs)


if __name__ == "__main__":
    main()
