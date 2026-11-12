import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import tensorflow as tf

from config import Config


def find_image_path(image_id: str, root_dir: str) -> str:
    """Cherche une image avec ou sans extension dans le dossier Test."""
    candidates = [image_id]
    for ext in [".jpg", ".jpeg", ".png"]:
        candidates.append(image_id + ext)
    for name in candidates:
        path = os.path.join(root_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image {image_id} introuvable dans {root_dir}")


def main():
    model = tf.keras.models.load_model("models/best_model.keras")

    # Charger les IDs du test
    test_df = pd.read_csv(str(Config.TEST_CSV))
    image_ids = test_df['ID'].astype(str).tolist()
    print(f"🖼️ Images à prédire : {len(image_ids)}")

    # Récupérer les noms de classes (sauvegardés à l'entraînement ou depuis le dossier Train)
    # ⚠️ À adapter : idéalement, sauvegarder class_names à l'entraînement dans un fichier JSON
    import json
    class_names_path = "models/class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path) as f:
            class_names = json.load(f)
    else:
        # Fallback : déduire depuis le dossier Train (si disponible en local)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(Config.TRAIN_DIR),
            image_size=Config.IMG_SIZE,
            batch_size=1,
        )
        class_names = train_ds.class_names

    predictions = []
    for image_id in image_ids:
        path = find_image_path(image_id, str(Config.TEST_DIR))
        img = tf.keras.utils.load_img(path, target_size=Config.IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr = tf.expand_dims(arr, axis=0)

        probs = model.predict(arr, verbose=0)[0]
        predicted_class = class_names[int(tf.argmax(probs))]
        predictions.append(predicted_class)

    # Créer le submission
    submission = pd.DataFrame({
        "ID": image_ids,
        "CLASS": predictions
    })
    submission.to_csv(str(Config.SUBMISSION_CSV), index=False)

    print(f"✅ {Config.SUBMISSION_CSV} créé avec {len(submission)} lignes")
    print(submission.head(10))


if __name__ == "__main__":
    main()