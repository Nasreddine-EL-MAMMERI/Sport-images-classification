"""Comparaison statistique des runs — référence gelée contre fine-tuning.

Usage :
    python -m src.compare_ft
    python -m src.compare_ft --arms ft30 ft100

Lit les probabilités sauvegardées par src.finetune (results/preds_*.npy)
et les labels (results/y_val.npy). Aucune image n'est rechargée.

Produit :
    results/comparison_finetuning.md   tableaux prêts à coller dans le README
    results/comparison_finetuning.png  deux graphiques

DEUX CHOSES QUE CE SCRIPT FAIT ET QU'UN SIMPLE « accuracy » NE FAIT PAS :

1. Test de McNemar. Les modèles prédisent LES MÊMES 2 007 images.
   Les comparer comme deux échantillons indépendants est une faute :
   leurs erreurs sont corrélées. McNemar ne regarde que les images
   DISCORDANTES — celles où un modèle a raison et l'autre tort.
       b = A juste, B faux
       c = A faux, B juste
   Sous H0 (aucune différence), b et c suivent une binomiale(0,5).

2. Intervalle de confiance bootstrap de la différence, apparié par image.
   Il répond à « si je tirais un autre jeu de validation de même taille,
   cet écart serait-il encore là ? ». Un écart dont l'IC contient 0 n'est
   pas un gain.

Deux tableaux sont produits : chaque bras contre la référence, puis les
bras entre eux. Sans le second, on ne peut pas dire si « 100 couches »
bat réellement « 30 couches », ou si les deux sont indiscernables.
"""

import argparse
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


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Test exact de McNemar sur les paires discordantes (A contre B)."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}

    try:
        from scipy.stats import binomtest

        p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
    except ImportError:  # scipy ancien
        from scipy.stats import binom_test

        p = float(binom_test(min(b, c), b + c, 0.5, alternative="two-sided"))

    return {"b": b, "c": c, "p_value": p}


def bootstrap_diff(correct_a: np.ndarray, correct_b: np.ndarray,
                   n_boot: int = N_BOOTSTRAP, seed: int = Config.SEED) -> tuple:
    """IC à 95 % de (exactitude_b - exactitude_a), apparié par image."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = correct_b[idx].mean() - correct_a[idx].mean()
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), diffs


def bootstrap_acc_ci(correct: np.ndarray, n_boot: int = N_BOOTSTRAP,
                     seed: int = Config.SEED) -> tuple:
    rng = np.random.default_rng(seed + 1)
    n = len(correct)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        accs[i] = correct[rng.integers(0, n, n)].mean()
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def compare_pair(name_a: str, correct_a: np.ndarray,
                 name_b: str, correct_b: np.ndarray) -> dict:
    acc_a = float(correct_a.mean())
    acc_b = float(correct_b.mean())
    lo, hi, diffs = bootstrap_diff(correct_a, correct_b)
    return {
        "a": name_a,
        "b": name_b,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "delta_pts": (acc_b - acc_a) * 100,
        "delta_ci": (lo * 100, hi * 100),
        "mcnemar": mcnemar(correct_a, correct_b),
        "n_discordant": int(np.sum(correct_a != correct_b)),
        "diffs": diffs,
    }


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
                f"{path} absent. Lancez d'abord `python -m src.finetune`."
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

    # ---------- Chargement de tous les runs ----------
    runs = {"phase1": ref_probs}
    for tag in tags:
        path = Config.RESULTS_DIR / f"preds_{tag}.npy"
        if not path.exists():
            print(f"⚠️  {path} absent — run ignoré")
            continue
        probs = np.load(path)
        if len(probs) != len(y_true):
            raise ValueError(f"{tag} : {len(probs)} prédictions pour "
                             f"{len(y_true)} labels")
        runs[tag] = probs

    correct = {name: p.argmax(1) == y_true for name, p in runs.items()}
    acc = {name: float(c.mean()) for name, c in correct.items()}

    # ---------- Tableau 1 : chaque bras contre la référence ----------
    vs_ref = [compare_pair("phase1", correct["phase1"], tag, correct[tag])
              for tag in runs if tag != "phase1"]

    # ---------- Tableau 2 : les bras entre eux ----------
    others = [t for t in runs if t != "phase1"]
    between = []
    for i in range(len(others)):
        for j in range(i + 1, len(others)):
            between.append(
                compare_pair(others[i], correct[others[i]],
                             others[j], correct[others[j]])
            )

    # ================= Console =================
    print("\n" + "=" * 84)
    print(f"Validation : {len(y_true)} images — tous les modèles sur les mêmes images")
    print("=" * 84)
    print(f"{'run':10s} {'exactitude':>11s} {'IC 95 % (exactitude)':>24s}")
    print("-" * 84)
    for name in runs:
        lo, hi = bootstrap_acc_ci(correct[name])
        print(f"{name:10s} {acc[name]:>11.4f}    [{lo:.4f} ; {hi:.4f}]")

    print("\n--- Chaque bras contre la référence gelée ---")
    print(f"{'paire':22s} {'écart (pt)':>11s} {'IC 95 % écart':>22s} "
          f"{'disc.':>6s} {'p':>7s}")
    print("-" * 84)
    for r in vs_ref:
        lo, hi = r["delta_ci"]
        print(f"{r['a'] + ' → ' + r['b']:22s} {r['delta_pts']:>+11.2f} "
              f"{lo:>+8.2f} / {hi:>+7.2f} {r['n_discordant']:>6d} "
              f"{r['mcnemar']['p_value']:>7.3f}")

    if between:
        print("\n--- Les bras entre eux ---")
        print(f"{'paire':22s} {'écart (pt)':>11s} {'IC 95 % écart':>22s} "
              f"{'disc.':>6s} {'p':>7s}")
        print("-" * 84)
        for r in between:
            lo, hi = r["delta_ci"]
            print(f"{r['a'] + ' → ' + r['b']:22s} {r['delta_pts']:>+11.2f} "
                  f"{lo:>+8.2f} / {hi:>+7.2f} {r['n_discordant']:>6d} "
                  f"{r['mcnemar']['p_value']:>7.3f}")

    print("\n--- Lecture ---")
    for r in vs_ref + between:
        mc = r["mcnemar"]
        if mc["p_value"] < 0.05:
            verdict = "différence significative au seuil 0,05"
        else:
            verdict = "AUCUNE différence démontrable"
        print(f"{r['a']} → {r['b']:10s} b={mc['b']:3d} / c={mc['c']:3d}  "
              f"→ {verdict}")

    # ================= Graphiques =================
    names = list(runs)
    accs = [acc[n] for n in names]
    errs = []
    for n in names:
        lo, hi = bootstrap_acc_ci(correct[n])
        errs.append([acc[n] - lo, hi - acc[n]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#8c8c8c"] + ["#2f6fba"] * (len(names) - 1)
    axes[0].bar(names, accs, color=colors[:len(names)])
    axes[0].errorbar(range(len(accs)), accs, yerr=np.array(errs).T,
                     fmt="none", ecolor="black", capsize=6, linewidth=1.2)
    axes[0].set_ylim(min(accs) - 0.02, min(1.0, max(accs) + 0.02))
    axes[0].set_ylabel("Exactitude de validation")
    axes[0].set_title(f"Exactitude (IC 95 % bootstrap, {len(y_true)} images)")
    for i, a in enumerate(accs):
        axes[0].text(i, a, f"{a:.4f}", ha="center", va="bottom", fontsize=9)

    if vs_ref:
        best = max(vs_ref, key=lambda r: r["delta_pts"])
        axes[1].hist(best["diffs"] * 100, bins=40, color="#2f6fba", alpha=0.75)
        axes[1].axvline(0, color="black", linestyle="--", linewidth=1,
                        label="aucun effet")
        axes[1].axvline(best["delta_pts"], color="#c0392b", linewidth=2,
                        label=f"écart observé {best['delta_pts']:+.2f} pt")
        axes[1].set_xlabel("Écart d'exactitude (points de %)")
        axes[1].set_ylabel("Tirages bootstrap")
        axes[1].set_title(f"Distribution de l'écart — phase1 → {best['b']}")
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    png_path = Config.RESULTS_DIR / "comparison_finetuning.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ {png_path}")

    # ================= Markdown =================
    def row(r):
        lo, hi = r["delta_ci"]
        return (
            f"| {r['a']} → {r['b']} | {r['acc_b']:.4f} | "
            f"{r['delta_pts']:+.2f} pt | [{lo:+.2f} ; {hi:+.2f}] | "
            f"{r['n_discordant']} | {r['mcnemar']['p_value']:.3f} |"
        )

    header = ("| Comparaison | Exactitude | Écart | IC 95 % de l'écart | "
              "Images discordantes | McNemar p |")
    sep = "|---|---|---|---|---|---|"

    lines = ["## Chaque bras contre la référence gelée", "",
             header, sep] + [row(r) for r in vs_ref]
    if between:
        lines += ["", "## Les bras entre eux", "", header, sep] + \
                 [row(r) for r in between]

    lines += [
        "",
        f"_Validation : {len(y_true)} images ; "
        f"IC bootstrap à 95 % sur {N_BOOTSTRAP} rééchantillonnages ; "
        f"McNemar exact sur les paires discordantes._",
    ]

    md_path = Config.RESULTS_DIR / "comparison_finetuning.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ {md_path}")


if __name__ == "__main__":
    main()
