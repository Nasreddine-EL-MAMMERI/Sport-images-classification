import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from src.data import get_datasets, get_augmentation
from src.model import build_model
from config import Config


def main():
    # ===== 1. Charger les données =====
    print("📦 Chargement des données...")
    train_ds, val_ds = get_datasets(
        str(Config.TRAIN_DIR),
        validation_split=Config.VALIDATION_SPLIT
    )

    # Détecter les classes automatiquement depuis le dossier Train
    class_names = train_ds.class_names
    Config.NUM_CLASSES = len(class_names)
    print(f"🏷️ Classes détectées ({Config.NUM_CLASSES}) : {class_names}")

    # ===== 2. Augmentation de données =====
    if Config.USE_AUGMENTATION:
        augmentation = get_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # ===== 3. Construire le modèle =====
    print("🧠 Construction du modèle...")
    model, base_model = build_model(
        num_classes=Config.NUM_CLASSES,
        img_size=Config.IMG_SIZE,
        fine_tune_at=Config.FINE_TUNE_AT
    )
    model.summary()

    # ===== 4. Callbacks =====
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    # ===== 5. Phase 1 : entraîner seulement la tête =====
    print("\n=== PHASE 1 : Tête de classification uniquement ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS_PHASE1,
        callbacks=callbacks,
    )

    # ===== 6. Phase 2 : Fine-tuning du backbone =====
    print("\n=== PHASE 2 : Fine-tuning du backbone ===")
    model.unfreeze_for_fine_tuning()

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS_PHASE2,
        callbacks=callbacks,
        initial_epoch=history1.epoch[-1] + 1,
    )

    # ===== 7. Sauvegarder =====
    os.makedirs("models", exist_ok=True)
    model.save("models/final_model.keras")
    print("\n✅ Modèle sauvegardé dans models/final_model.keras")


if __name__ == "__main__":
    main()