"""Phase 2 — fine-tuning du backbone, pour COMPARAISON avec la phase 1.

Ce script ne remplace rien : il part du modèle gelé déjà entraîné
(`models/best_model.keras`, val 0,95665 à l'époque 14), dégèle les N
dernières couches du backbone, et réentraîne à learning rate réduit.
Le modèle de référence n'est jamais écrasé.

Usage (depuis la racine du projet) :
    python -m src.finetune --unfreeze 100
    python -m src.finetune --unfreeze 30

Produit :
    models/best_model_ft{N}.keras      meilleur modèle de la phase 2
    models/history_ft{N}.json          courbes, lr, temps par époque
    results/preds_phase1.npy           probas du modèle gelé sur val (1 fois)
    results/preds_ft{N}.npy            probas du modèle fine-tuné sur val
    results/summary_ft{N}.json         résumé chiffré de l'expérience

DESIGN EXPÉRIMENTAL — une seule variable change à la fois :
    le nombre de couches dégelées (et le lr, indissociable).
    Tout le reste (split, seed, batch 32, callbacks, 224×224) est identique
    à la phase 1. Sinon la comparaison ne mesure plus rien.
"""

import argparse
import json
import os
import sys
import time

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


# =====================================================================
# 1. Localiser le backbone dans le modèle rechargé
# =====================================================================
def find_backbone(model):
    """Retourne le sous-modèle EfficientNetB0 imbriqué dans `model`.

    Après `load_model`, le backbone est un modèle fonctionnel imbriqué.
    On le cherche par nom, avec un repli sur le plus gros sous-modèle.
    """
    for layer in model.layers:
        if "efficientnet" in layer.name.lower() and isinstance(layer, tf.keras.Model):
            return layer

    nested = [l for l in model.layers if isinstance(l, tf.keras.Model)]
    if not nested:
        raise RuntimeError(
            "Aucun sous-modèle trouvé : le backbone n'est pas imbriqué dans "
            f"ce fichier. Couches : {[l.name for l in model.layers]}"
        )
    return max(nested, key=lambda m: m.count_params())


def count_params(model) -> tuple[int, int]:
    total = int(model.count_params())
    trainable = int(sum(
        int(np.prod(w.shape)) for w in model.trainable_weights
    ))
    return total, trainable


# =====================================================================
# 2. Dégeler les N dernières couches
# =====================================================================
def unfreeze_last(base_model, n_layers: int, learning_rate: float):
    """Dégèle les `n_layers` dernières couches du backbone, BN exclues.

    Deux précautions non négociables :
      - RE-COMPILER : Keras conserve sinon l'ancien graphe et rien ne bouge.
      - Garder les BatchNorm du backbone gelées : avec des lots de 32 et un
        domaine différent d'ImageNet, leurs statistiques se recalculent mal
        et écrasent les poids pré-entraînés.

    Note : dans `build_model`, l'appel au backbone est écrit
    `base_model(inputs, training=False)`. Ce drapeau est figé dans le graphe
    sauvegardé. Conséquence voulue ici : même dégelé, le backbone tourne en
    mode inférence, donc ses BatchNorm n'accumulent jamais de nouvelles
    statistiques. C'est exactement le comportement recommandé.
    """
    total_layers = len(base_model.layers)
    if not 1 <= n_layers <= total_layers:
        raise ValueError(f"n_layers doit être dans [1, {total_layers}]")

    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    frozen_bn = 0
    for layer in base_model.layers[-n_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1

    n_unfrozen = sum(
        1 for layer in base_model.layers[-n_layers:] if layer.trainable
    )
    print(f"🔓 {n_unfrozen} couches dégelées sur {total_layers} "
          f"({frozen_bn} BatchNorm maintenues gelées) — lr={learning_rate}")
    return base_model


# =====================================================================
# 3. Pipeline tf.data (mémoire : on ne convertit pas 8 024 images d'un coup)
# =====================================================================
def make_dataset(X, y, batch_size, shuffle, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, t: (preprocess_input(tf.cast(x, tf.float32)), t),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# =====================================================================
# 4. Principal
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unfreeze", type=int, default=Config.FINE_TUNE_AT,
                        help="nombre de couches dégelées en partant de la fin")
    parser.add_argument("--epochs", type=int, default=15,
                        help="époque maximale de la phase 2")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE_PHASE2)
    parser.add_argument("--smoke", action="store_true",
                        help="1 époque sur un sous-jeu, pour tester le script")
    args = parser.parse_args()

    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(Config.SEED)

    # ---- Données (identiques à la phase 1 : même split, même seed) ----
    data = load_datasets(Config, load_test=False)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    # Sauvegardé une fois : permet à src.compare_ft de tourner sans
    # recharger les 10 031 images (utile dans un notebook séparé).
    if not args.smoke:
        np.save(Config.RESULTS_DIR / "y_val.npy", y_val)
        (Config.RESULTS_DIR / "class_names.json").write_text(
            json.dumps(data["class_names"], ensure_ascii=False), encoding="utf-8"
        )

    if args.smoke:
        X_train, y_train = X_train[:640], y_train[:640]
        X_val, y_val = X_val[:320], y_val[:320]
        args.epochs = 1
        print("🧪 Mode smoke : 640 images, 1 époque.")

    # ---- Modèle de référence : NON modifié, NON écrasé ----
    ref_path = Config.BEST_MODEL_PATH
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} introuvable. Lancez d'abord `python -m src.train`."
        )
    model = tf.keras.models.load_model(str(ref_path))
    base_model = find_backbone(model)
    print(f"📦 Référence chargée : {ref_path} "
          f"({base_model.name}, {len(base_model.layers)} couches)")

    def val_predictions(m):
        ds = make_dataset(X_val, y_val, Config.BATCH_SIZE, shuffle=False)
        return m.predict(ds, verbose=0)

    tag = f"ft{args.unfreeze}" + ("_smoke" if args.smoke else "")

    # Référence mesurée une seule fois, puis mise en cache.
    ref_preds_path = Config.RESULTS_DIR / "preds_phase1.npy"
    # En mode smoke on ne met RIEN en cache : les prédictions portent sur
    # 320 images et corrompraient la comparaison réelle (2 007 images).
    if ref_preds_path.exists() and not args.smoke:
        ref_probs = np.load(ref_preds_path)
        print(f"♻️  Prédictions de référence rechargées : {ref_preds_path}")
    else:
        ref_probs = val_predictions(model)
        if not args.smoke:
            np.save(ref_preds_path, ref_probs)

    ref_acc = float((ref_probs.argmax(1) == y_val.argmax(1)).mean())
    print(f"🎯 Exactitude de référence mesurée : {ref_acc:.4f}")

    # ---- Dégel + recompilation ----
    total_before, trainable_before = count_params(model)
    unfreeze_last(base_model, args.unfreeze, args.lr)
    total_after, trainable_after = count_params(model)
    print(f"🔢 Paramètres entraînables : {trainable_before:,} → "
          f"{trainable_after:,} sur {total_after:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ---- Callbacks : mêmes valeurs que la phase 1 ----
    ckpt_path = Config.MODELS_DIR / f"best_model_{tag}.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
        ),
        ModelCheckpoint(
            filepath=str(ckpt_path),
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

    train_ds = make_dataset(X_train, y_train, Config.BATCH_SIZE,
                            shuffle=True, seed=Config.SEED)
    val_ds = make_dataset(X_val, y_val, Config.BATCH_SIZE, shuffle=False)

    print(f"\n=== PHASE 2 : {args.unfreeze} couches dégelées, "
          f"{args.epochs} époques max, lr={args.lr} ===")
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    duration = time.time() - start

    # ---- Résultats de la phase 2 ----
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist["unfreeze"] = args.unfreeze
    hist["learning_rate_phase2"] = args.lr
    hist["duration_seconds"] = round(duration, 1)
    (Config.MODELS_DIR / f"history_{tag}.json").write_text(
        json.dumps(hist, indent=2), encoding="utf-8"
    )

    ft_probs = val_predictions(model)
    np.save(Config.RESULTS_DIR / f"preds_{tag}.npy", ft_probs)
    ft_acc = float((ft_probs.argmax(1) == y_val.argmax(1)).mean())
    best_epoch = int(np.argmax(hist["val_accuracy"])) + 1
    best_val = float(np.max(hist["val_accuracy"]))

    summary = {
        "unfreeze_layers": args.unfreeze,
        "learning_rate_phase2": args.lr,
        "epochs_run": len(hist["loss"]),
        "duration_minutes": round(duration / 60, 1),
        "params_total": total_after,
        "params_trainable_phase1": trainable_before,
        "params_trainable_phase2": trainable_after,
        "reference_val_accuracy": round(ref_acc, 5),
        "phase2_best_val_accuracy": round(best_val, 5),
        "phase2_best_epoch": best_epoch,
        "phase2_final_val_accuracy": round(ft_acc, 5),
        "delta_points": round((ft_acc - ref_acc) * 100, 3),
        "agreement_with_reference": round(
            float((ft_probs.argmax(1) == ref_probs.argmax(1)).mean()), 4
        ),
        "n_val": int(len(y_val)),
    }
    (Config.RESULTS_DIR / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print(f"RÉFÉRENCE (gelé)      : {ref_acc:.4f}")
    print(f"PHASE 2 (dégelé ×{args.unfreeze:<3}) : {ft_acc:.4f}   "
          f"→ écart {(ft_acc - ref_acc)*100:+.2f} pt")
    print(f"Meilleure val phase 2 : {best_val:.4f} (époque {best_epoch})")
    print(f"Accord des prédictions: {summary['agreement_with_reference']*100:.1f}%")
    print(f"Durée                 : {duration/60:.1f} min")
    print(f"Résumé                : {Config.RESULTS_DIR / f'summary_{tag}.json'}")
    print("=" * 62)
    print("\n⚠️  Un écart inférieur à ~1 point sur 2 007 images n'est pas "
          "démontrable :\n    lancez src.compare_ft pour le test de McNemar "
          "et l'intervalle de confiance.")


if __name__ == "__main__":
    main()
