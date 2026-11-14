"""Figures de synthèse pour le README — fine-tuning d'EfficientNetB0.

ATTENTION : ce script ne MESURE rien. Il transcrit des valeurs déjà mesurées,
sources ci-dessous, et les met en forme. Le script de mesure est
`src/compare_ft.py` ; celui-ci ne sert qu'à la présentation.

Sources des valeurs :
  - results/summary_ft30.json, results/summary_ft100.json  (validation, params)
  - results/comparison_finetuning.md                        (McNemar, IC bootstrap)
  - page Submissions de la compétition                      (public, privé)
  - classement final privé de la compétition                (positions 1 à 13)

Usage :
    python -m src.plot_summary
"""

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config import Config

# =====================================================================
# Valeurs mesurées
# =====================================================================
MODELS = ["Frozen\n(349,769 trainable)",
          "Fine-tune 30\n(1,833,129 trainable)",
          "Fine-tune 100\n(3,798,909 trainable)"]
COLORS = ["#9a9a9a", "#7aa6d6", "#1f4e79"]

# (correct, total) — les scores Kaggle se reconstituent en entiers
SPLITS = {
    "Validation\n(2,007 images)": [(1919, 2007), (1927, 2007), (1934, 2007)],
    "Public LB\n(557 images)":    [(518, 557), (522, 557), (522, 557)],
    "Private LB\n(558 images)":   [(512, 558), (522, 558), (522, 558)],
}

# Comparaisons appariées : (b = A juste / B faux, c = A faux / B juste, p)
PAIRS = [
    ("frozen → ft30", 3, 11, 0.057, +0.40, (+0.05, +0.80)),
    ("frozen → ft100", 4, 19, 0.003, +0.75, (+0.30, +1.25)),
    ("ft30 → ft100", 13, 6, 0.167, -0.35, (-0.80, +0.10)),
]

# Classement privé final
RANKS = [0.95878, 0.95878, 0.94982, 0.94982, 0.93906, 0.93906, 0.93369,
         0.92831, 0.92473, 0.92473, 0.92114, 0.92114, 0.91577]

PRIVATE_FROZEN = 512 / 558
PRIVATE_FT = 522 / 558


def fig_main():
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
    fig.suptitle(
        "EfficientNetB0 on 73 sports classes — frozen backbone vs fine-tuning",
        fontsize=15, fontweight="bold", y=0.985,
    )

    # ---------- A. Exactitude par jeu ----------
    ax = axes[0, 0]
    labels = list(SPLITS)
    x = np.arange(len(labels))
    width = 0.26
    for i, model in enumerate(MODELS):
        accs = [SPLITS[s][i][0] / SPLITS[s][i][1] for s in labels]
        bars = ax.bar(x + (i - 1) * width, accs, width, label=model.split("\n")[0],
                      color=COLORS[i], edgecolor="white", linewidth=0.8)
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.002, f"{a:.4f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.90, 0.992)
    ax.set_ylabel("Accuracy")
    ax.set_title("A · Accuracy is consistent across three independent splits",
                 fontsize=11, loc="left")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=1.0,
              edgecolor="0.8", ncol=3, borderpad=0.4)
    ax.grid(axis="y", alpha=0.25)

    # ---------- B. Écarts avec IC 95 % ----------
    ax = axes[0, 1]
    names = [p[0] for p in PAIRS]
    ypos = np.arange(len(PAIRS))[::-1]
    for y, (name, b, c, p, delta, ci) in zip(ypos, PAIRS):
        sig = p < 0.05
        color = "#1f4e79" if sig else "#b0b0b0"
        ax.plot([ci[0], ci[1]], [y, y], color=color, linewidth=6,
                solid_capstyle="round", alpha=0.85)
        ax.plot(delta, y, "o", color=color, markersize=11,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.text(ci[1] + 0.04, y, f"p = {p:.3f}" + ("  ✓" if sig else ""),
                va="center", fontsize=9, color=color, fontweight="bold" if sig else "normal")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(-1.1, 1.75)
    ax.set_xlabel("Δ validation accuracy (percentage points), 95 % bootstrap CI")
    ax.set_title("B · Fine-tuning helps; the depth does not",
                 fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.25)

    # ---------- C. Paires discordantes (McNemar) ----------
    ax = axes[1, 0]
    ypos = np.arange(len(PAIRS))[::-1]
    for y, (name, b, c, p, delta, ci) in zip(ypos, PAIRS):
        ax.barh(y + 0.17, -b, height=0.32, color="#c0392b",
                edgecolor="white", label="first model right" if y == ypos[0] else None)
        ax.barh(y - 0.17, c, height=0.32, color="#27ae60",
                edgecolor="white", label="second model right" if y == ypos[0] else None)
        ax.text(-b - 0.4, y + 0.17, str(b), va="center", ha="right", fontsize=9)
        ax.text(c + 0.4, y - 0.17, str(c), va="center", ha="left", fontsize=9)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlim(-17, 25)
    ax.set_xlabel("images where exactly one of the two models is correct")
    ax.set_title("C · McNemar looks only at the images that flipped",
                 fontsize=11, loc="left")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)

    # ---------- D. Position au classement privé ----------
    ax = axes[1, 1]
    for i, score in enumerate(RANKS):
        rank = i + 1
        offset = -0.16 if i > 0 and RANKS[i - 1] == score else 0.0
        ax.plot(score, rank + offset, "o", color="#555", markersize=8,
                markeredgecolor="white", markeredgewidth=1)
        ax.text(score - 0.0009, rank + offset, f"#{rank}", ha="right",
                va="center", fontsize=8, color="#555")

    ax.axvline(PRIVATE_FT, color="#1f4e79", linewidth=2.2, linestyle="-")
    ax.axvline(PRIVATE_FROZEN, color="#9a9a9a", linewidth=1.8, linestyle=":")

    ax.annotate(
        "this project, fine-tuned\n0.93548 — not ranked\n(would place 7th)",
        xy=(PRIVATE_FT, 11.6), xytext=(PRIVATE_FT + 0.0022, 12.5),
        fontsize=8.5, color="#1f4e79", fontweight="bold",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.3),
    )
    ax.annotate(
        "before fine-tuning\n0.91756",
        xy=(PRIVATE_FROZEN, 3.0), xytext=(PRIVATE_FROZEN + 0.0016, 2.2),
        fontsize=8, color="#666",
        arrowprops=dict(arrowstyle="->", color="#9a9a9a", lw=1.1),
    )

    ax.set_xlim(0.9105, 0.9625)
    ax.set_ylim(13.6, 0.4)
    ax.set_yticks(range(1, 14))
    ax.set_ylabel("Final private rank")
    ax.set_xlabel("Private leaderboard score")
    ax.set_title("D · Where 0.93548 sits in the closed leaderboard",
                 fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = Config.RESULTS_DIR / "finetuning_comparison.png"
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ {out}")


def fig_gain():
    """Le gain en points de pourcentage, comparable entre jeux de tailles différentes."""
    fig, ax = plt.subplots(figsize=(10, 4.4))

    labels = ["Validation", "Public LB", "Private LB"]
    deltas = [0.75, 0.72, 1.79]
    imgs = [15, 4, 10]
    totals = [2007, 557, 558]
    colors = ["#7aa6d6", "#7aa6d6", "#1f4e79"]

    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, deltas, height=0.5, color=colors, edgecolor="white")
    for b, d, n, t in zip(bars, deltas, imgs, totals):
        ax.text(d + 0.04, b.get_y() + b.get_height() / 2,
                f"+{d:.2f} pt   (+{n} images of {t:,})",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 2.55)
    ax.set_xlabel("Accuracy gain after fine-tuning, in percentage points")
    ax.set_title("The same improvement, measured on three independent splits",
                 fontsize=12, loc="left")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    out = Config.RESULTS_DIR / "finetuning_gain.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\u2705 {out}")


if __name__ == "__main__":
    fig_main()
    fig_gain()
