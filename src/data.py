import tensorflow as tf
from config import Config


def get_datasets(train_dir: str, validation_split: float = 0.2):
    """
    Charge les images depuis train_dir et sépare automatiquement
    en training et validation grâce à validation_split de Keras.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMG_SIZE,
        shuffle=True,
        seed=Config.SEED,
        validation_split=validation_split,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMG_SIZE,
        shuffle=False,
        seed=Config.SEED,
        validation_split=validation_split,
        subset="validation",
    )

    # Mise en cache + prefetch pour la performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


def get_augmentation():
    """Pipeline d'augmentation de données (Keras layers)."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=Config.SEED),
        tf.keras.layers.RandomRotation(0.1, seed=Config.SEED),
        tf.keras.layers.RandomZoom(0.1, seed=Config.SEED),
        tf.keras.layers.RandomContrast(0.1, seed=Config.SEED),
    ], name="data_augmentation")


def get_test_dataset(test_dir: str):
    """Charge le dossier Test (sans labels) pour la prédiction finale."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode=None,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMG_SIZE,
        shuffle=False,
    )
    return test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)