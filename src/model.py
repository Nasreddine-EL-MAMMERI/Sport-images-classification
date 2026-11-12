import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from config import Config


def build_model(num_classes: int, img_size=(224, 224), fine_tune_at: int = 100):
    """
    Construit un modèle de fine-tuning :
    1. Backbone EfficientNetB0 pré-entraîné sur ImageNet (gelé au début)
    2. Tête de classification personnalisée
    3. Possibilité de débloquer le backbone pour fine-tuning
    """
    # Backbone pré-entraîné (sans la couche de classification finale)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )

    # Phase 1 : geler le backbone, entraîner seulement la tête
    base_model.trainable = False

    # Ajouter nos propres couches
    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)  # training=False pour BatchNorm en inférence
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="efficientnet_b0_finetune")

    # Compilation phase 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LR_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fonction pour débloquer le fine-tuning (phase 2)
    def unfreeze_for_fine_tuning():
        base_model.trainable = True
        # Geler les couches du début, débloquer seulement la fin
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        # Re-compiler avec un learning rate plus petit
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LR_PHASE2),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    model.unfreeze_for_fine_tuning = unfreeze_for_fine_tuning
    model.base_model = base_model

    return model, base_model