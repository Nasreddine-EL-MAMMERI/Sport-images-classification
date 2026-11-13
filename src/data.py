"""Chargement des données — reprise du pipeline NumPy du notebook.

  - chargement NumPy + PIL au lieu de image_dataset_from_directory ;
  - split STRATIFIÉ (train_test_split, stratify=CLASS) au lieu de
    validation_split, qui n'est pas stratifié ;

"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from config import Config


def make_path(image_id: str, images_dir) -> str | None:
    """Retrouve le chemin d'une image en testant les extensions courantes."""
    for ext in (".jpg", ".jpeg", ".png", ""):
        path = os.path.join(str(images_dir), str(image_id) + ext)
        if os.path.exists(path):
            return path
    return None


def add_filepaths(df: pd.DataFrame, images_dir) -> pd.DataFrame:
    """Ajoute une colonne `filepath`, en signalant les images manquantes."""
    df = df.copy()
    df["filepath"] = df["ID"].apply(lambda x: make_path(x, images_dir))

    missing = int(df["filepath"].isna().sum())
    print(f"❌ Images manquantes : {missing} / {len(df)}")
    if missing:
        print("   Exemples :", df.loc[df["filepath"].isna(), "ID"].head().tolist())

    return df


def load_images(
    df: pd.DataFrame,
    img_size=Config.IMG_SIZE,
    with_labels: bool = True,
    label_to_idx: dict | None = None,
):
    """Charge les images en uint8 (0-255), comme le notebook.

    EfficientNetB0 de Keras attend des pixels bruts 0-255 : la normalisation
    est incluse dans le modèle. Ne PAS diviser par 255.

    Returns:
        images (N, H, W, 3) uint8
        labels (N, C) one-hot, ou None si with_labels=False
        positions : indices des lignes effectivement chargées
    """
    images, labels, positions = [], [], []

    for position, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                             desc="Chargement")):
        try:
            img = Image.open(row["filepath"]).convert("RGB")
            img = img.resize(img_size)
            arr = np.array(img, dtype=np.uint8)

            label = None
            if with_labels:
                # Lève un KeyError si la colonne CLASS est absente : c'est
                # exactement le bug qui faisait compter les images de test
                # comme des erreurs dans le notebook.
                label = label_to_idx[row["CLASS"]]

            images.append(arr)
            positions.append(position)
            if with_labels:
                labels.append(label)
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {row.get('ID', '?')} : {exc}")
            continue

    if not images:
        raise ValueError("Aucune image chargée : vérifiez IMG_DIR et les CSV.")

    images = np.stack(images)
    print(f"✅ {len(images)} images chargées — shape {images.shape}, "
          f"pixels [{images.min()}, {images.max()}]")

    if with_labels:
        labels = np.array(labels)
        return (
            images,
            to_categorical(labels, num_classes=len(label_to_idx)),
            np.array(positions),
        )

    return images, None, np.array(positions)


def load_datasets(cfg=Config, load_test: bool = True) -> dict:
    """Charge les CSV, construit les chemins et renvoie les tableaux prêts.

    Returns un dictionnaire : X_train, y_train, X_val, y_val,
    X_test, test_ids, class_names, n_classes.
    """
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    class_names = sorted(train_df["CLASS"].unique().tolist())
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"🏷️ {len(class_names)} classes détectées")

    train_df = add_filepaths(train_df, cfg.IMG_DIR)
    train_df = train_df.dropna(subset=["filepath"]).reset_index(drop=True)

    print("📦 Chargement des images d'entraînement…")
    images, labels, _ = load_images(
        train_df, cfg.IMG_SIZE, with_labels=True, label_to_idx=label_to_idx
    )

    # Split STRATIFIÉ : mêmes proportions de classes dans les deux jeux.
    indices = np.arange(len(images))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=cfg.VALIDATION_SPLIT,
        stratify=labels.argmax(axis=1),
        random_state=cfg.SEED,
    )

    result = {
        "X_train": images[train_idx],
        "y_train": labels[train_idx],
        "X_val": images[val_idx],
        "y_val": labels[val_idx],
        "X_test": None,
        "test_ids": None,
        "class_names": class_names,
        "n_classes": len(class_names),
    }

    print(f"📊 Train : {result['X_train'].shape[0]} | "
          f"Validation : {result['X_val'].shape[0]}")

    if load_test:
        test_df = pd.read_csv(cfg.TEST_CSV)
        test_df = add_filepaths(test_df, cfg.IMG_DIR)
        test_df = test_df.dropna(subset=["filepath"]).reset_index(drop=True)

        print("📦 Chargement des images de test…")
        X_test, _, _ = load_images(test_df, cfg.IMG_SIZE, with_labels=False)
        result["X_test"] = X_test
        result["test_ids"] = test_df["ID"].astype(str).tolist()
        print(f"📊 Test : {X_test.shape[0]}")

    return result
