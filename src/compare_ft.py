"""Comparaison statistique des runs — phase 1 (gelé) contre phase 2 (fine-tuné).

Usage :
    python -m src.compare_ft
    python -m src.compare_ft --arms ft30 ft100

Lit les probabilités sauvegardées par src.finetune (results/preds_*.npy)
et les labels (results/y_val.npy). Aucune image n'est rechargée.

Produit :
    results/comparison_finetuning.md   tableau prêt à coller dans le README
    results/comparison_finetuning.png  deux graphiques

DEUX CHOSES QUE CE SCRIPT FAIT ET QU'UN SIMPLE « accuracy » NE FAIT PAS :

1. Test de McNemar. Les deux modèles prédisent LES MÊMES 2 007 images.
   Les comparer comme deux échantillons indépendants est une faute :
   leurs erreurs sont corrélées. McNemar ne regarde que les images
   DISCORDANTES — celles où un modèle a raison et l'autre tort.
       b = référence juste, fine-tuning faux
       c = référence faux, fine-tuning juste
   Sous H0 (aucune différence), b et c suivent une binomiale(0,5).

2. Intervalle de confiance bootstrap de la différence. Il répond à
   « si je tirais un autre jeu de validation de même taille, cet écart
   serait-il encore là ? ». Un écart dont l'IC contient 0 n'est pas un gain.
"""

import argparse
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

N_BOOTSTRAP = 10_000


def mcnemar(ref_correct: np.ndarray, ft_correct: np.ndarray) -> dict:
    """Test exact de McNemar sur les paires discordantes."""
    b = int(np.sum(ref_correct & ~ft_correct))
    c = int(np.sum(~ref_correct & ft_correct))

    if b + c == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}

    try:
        from scipy.stats import binomtest

        p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    except ImportError:  # scipy ancien
        from scipy.stats import binom_test

        p = float(binom_test(min(b, c), b + c, 0.5, alternative="two-sided"))

    return {"b": b, "c": c, "p_value": p}


def bootstrap_diff(ref_correct: np.ndarray, ft_correct: np.ndarray,
                   n_boot: int = N_BOOTSTRAP, seed: int = Config.SEED) -> tuple:
    """IC à 95 % de (exactitude_ft - exactitude_ref), apparié par image."""
    rng = np.random.default_rng(seed)
    n = len(ref_correct)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = ft_correct[idx].mean() - ref_correct[idx].mean()
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), diffs


def bootstrap_acc_ci(correct: np.ndarray, n_boot: int = N_BOOTSTRAP,
                     seed: int = Config.SEED) -> tuple:
    rng = np.random.default_rng(seed + 1)
    n = len(correct)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        accs[i] = correct[rng.integers(0, n, n)].mean()
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=None,
                        help="tags à comparer, ex. ft30 ft100")
    args = parser.parse_args()

    y_val_path = Config.RESULTS_DIR / "y_val.npy"
    ref_path = Config.RESULTS_DIR / "preds_phase1.npy"

    for path in (y_val_path, ref_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} absent. Lancez d'abord `python -m src.finetune "
                f"--unfreeze 100` (il crée aussi la référence)."
            )

    y_true = np.load(y_val_path).argmax(axis=1)
    ref_probs = np.load(ref_path)

    if args.arms:
        tags = args.arms
    else:
        tags = sorted(
            p.stem.replace("preds_", "")
            for p in Config.RESULTS_DIR.glob("preds_ft*.npy")
        )
    if not tags:
        raise FileNotFoundError(
            "Aucun results/preds_ft*.npy : la phase 2 n'a pas encore tourné."
        )

    ref_correct = ref_probs.argmax(1) == y_true
    rows = []

    for tag in tags:
        path = Config.RESULTS_DIR / f"preds_{tag}.npy"
        if not path.exists():
            print(f"⚠️  {path} absent — run ignoré")
            continue

        probs = np.load(path)
        if len(probs) != len(y_true):
            raise ValueError(f"{tag} : {len(probs)} prédictions pour "
                             f"{len(y_true)} labels")

        ft_correct = probs.argmax(1) == y_true
        acc_ref = float(ref_correct.mean())
        acc_ft = float(ft_correct.mean())
        mc = mcnemar(ref_correct, ft_correct)
        lo, hi, diffs = bootstrap_diff(ref_correct, ft_correct)
        acc_lo, acc_hi = bootstrap_acc_ci(ft_correct)

        rows.append({
            "tag": tag,
            "acc": acc_ft,
            "acc_ci": (acc_lo, acc_hi),
            "delta_pts": (acc_ft - acc_ref) * 100,
            "delta_ci": (lo * 100, hi * 100),
            "mcnemar": mc,
            "agreement": float((probs.argmax(1) == ref_probs.argmax(1)).mean()),
            "diffs": diffs,
        })

    acc_ref = float(ref_correct.mean())
    ref_lo, ref_hi = bootstrap_acc_ci(ref_correct)

    # ================= Affichage console =================
    print("\n" + "=" * 78)
    print(f"Jeu de validation : {len(y_true)} images — référence "
          f"{acc_ref:.4f} [{ref_lo:.4f}, {ref_hi:.4f}]")
    print("=" * 78)
    print(f"{'run':8s} {'exactitude':>12s} {'écart (pt)':>12s} "
          f"{'IC 95 % écart':>20s} {'McNemar p':>11s}")
    print("-" * 78)

    for r in rows:
        lo, hi = r["delta_ci"]
        print(f"{r['tag']:8s} {r['acc']:>12.4f} {r['delta_pts']:>+12.2f} "
              f"{lo:>+8.2f} / {hi:>+7.2f}   {r['mcnemar']['p_value']:>9.3f}")

    print("-" * 78)
    for r in rows:
        mc = r["mcnemar"]
        verdict = ("différence significative au seuil 0,05"
                   if mc["p_value"] < 0.05
                   else "AUCUNE différence démontrable")
        print(f"{r['tag']:8s} b={mc['b']:4d} (réf juste / ft faux)   "
              f"c={mc['c']:4d} (réf faux / ft juste)   → {verdict}")

    # ================= Graphiques =================
    labels = ["phase 1\n(gelé)"] + [r["tag"] for r in rows]
    accs = [acc_ref] + [r["acc"] for r in rows]
    errs = [[acc_ref - ref_lo, ref_hi - acc_ref]] + [
        [r["acc"] - r["acc_ci"][0], r["acc_ci"][1] - r["acc"]] for r in rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(labels, accs, color=["#8c8c8c"] + ["#2f6fba"] * len(rows))
    axes[0].errorbar(range(len(accs)), accs,
                     yerr=np.array(errs).T, fmt="none", ecolor="black",
                     capsize=6, linewidth=1.2)
    axes[0].set_ylim(min(accs) - 0.02, min(1.0, max(accs) + 0.02))
    axes[0].set_ylabel("Exactitude de validation")
    axes[0].set_title(f"Exactitude (IC 95 % bootstrap, {len(y_true)} images)")
    for i, a in enumerate(accs):
        axes[0].text(i, a, f"{a:.4f}", ha="center", va="bottom", fontsize=9)

    if rows:
        r = rows[0]
        axes[1].hist(r["diffs"] * 100, bins=40, color="#2f6fba", alpha=0.75)
        axes[1].axvline(0, color="black", linestyle="--", linewidth=1,
                        label="aucun effet")
        axes[1].axvline(r["delta_pts"], color="#c0392b", linewidth=2,
                        label=f"écart observé {r['delta_pts']:+.2f} pt")
        axes[1].set_xlabel("Écart d'exactitude (points de %)")
        axes[1].set_ylabel("Tirages bootstrap")
        axes[1].set_title(f"Distribution de l'écart — {r['tag']}")
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    png_path = Config.RESULTS_DIR / "comparison_finetuning.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ {png_path}")

    # ================= Markdown =================
    lines = [
        "| Run | Exactitude validation | Écart vs phase 1 | IC 95 % de l'écart | McNemar p | Accord des prédictions |",
        "|---|---|---|---|---|---|",
        f"| phase 1 (gelé) | {acc_ref:.4f} | — | — | — | — |",
    ]
    for r in rows:
        lo, hi = r["delta_ci"]
        lines.append(
            f"| {r['tag']} | {r['acc']:.4f} | {r['delta_pts']:+.2f} pt | "
            f"[{lo:+.2f} ; {hi:+.2f}] | {r['mcnemar']['p_value']:.3f} | "
            f"{r['agreement']*100:.1f} % |"
        )
    lines += [
        "",
        f"_Validation : {len(y_true)} images ; IC bootstrap à 95 %, "
        f"{N_BOOTSTRAP} rééchantillonnages ; McNemar exact sur les paires "
        f"discordantes._",
    ]
    md_path = Config.RESULTS_DIR / "comparison_finetuning.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ {md_path}")


if __name__ == "__main__":
    main()
