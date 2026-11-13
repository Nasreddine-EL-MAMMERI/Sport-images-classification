"""Modèle EfficientNetB0 + tête de classification.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from config import Config


def build_model(
    num_classes: int,
    img_size=Config.IMG_SIZE,
    learning_rate: float = Config.LEARNING_RATE_PHASE1,
):
    """Construit le modèle : backbone gelé + tête de classification.

    Returns:
        (model, base_model)
    """
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3))
    # training=False fige les BatchNorm du backbone en mode inférence :
    # indispensable tant que le backbone n'est pas dégelé.
    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnet_b0_feature_extraction")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_for_fine_tuning(
    model,
    base_model,
    fine_tune_at: int = Config.FINE_TUNE_AT,
    learning_rate: float = Config.LEARNING_RATE_PHASE2,
):
    """Dégèle les `fine_tune_at` dernières couches du backbone et recompile.

    Deux précautions qui décident du succès :
      1. RE-COMPILER après avoir changé `trainable`, sinon Keras conserve
         l'ancien graphe et l'entraînement ne change rien.
      2. Garder les BatchNorm du backbone en mode inférence : avec des lots
         de 32, leurs statistiques se recalculent mal et détruisent les poids
         pré-entraînés.
    """
    base_model.trainable = True

    for layer in base_model.layers[:-fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    n_trainable = sum(1 for layer in base_model.layers if layer.trainable)
    print(f"🔓 Backbone dégelé : {n_trainable}/{len(base_model.layers)} "
          f"couches entraînables, lr={learning_rate}")
    return model


