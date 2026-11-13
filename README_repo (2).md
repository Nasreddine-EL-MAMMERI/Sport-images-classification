# Sport Image Classification — Transfer Learning with EfficientNetB0

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-private%200.9176-20BEFF)
![Classes](https://img.shields.io/badge/classes-73-lightgrey)

Classifying sports images into **73 classes** (air hockey, basketball, curling,
figure skating, …) using an ImageNet-pretrained **EfficientNetB0** backbone with
a custom classification head.

| Metric | Value |
|---|---|
| Validation accuracy (2,007 images) | **0.9567** |
| Kaggle private leaderboard | **0.91756** |
| Kaggle public leaderboard | **0.92998** |
| Trainable parameters | 349,769 / 4,402,412 |
| Training time | ~6 min (20 epochs, Tesla T4) |

> **Note on ranking.** The competition closed before this submission was made,
> so it is marked *"Complete (after deadline)"* and does not appear in the
> final standings. Both public and private scores are reported — no rank is
> claimed.

> **Note on naming.** The backbone is **frozen**: this is *feature extraction*,
> not fine-tuning. A `unfreeze_for_fine_tuning()` function is provided in
> `src/model.py` but is **disabled by default** (`RUN_FINE_TUNING = False`),
> because the scores above were obtained without it. Enabling it would require
> re-training and re-measuring.

---

## The problem

- 73 classes, ~140 images per class on average
- 10,031 labelled training images, 1,115 test images
- No separate validation set: one had to be carved out of the training data
- Images of varying sizes

## Why transfer learning

I first trained a CNN from scratch. It reached **~25 % accuracy** and remained
unstable: with 73 classes and ~10,000 images, there is not enough signal to
learn rich visual features from random initialisation.

Switching to an ImageNet-pretrained EfficientNetB0 — frozen — and training only
a new head took validation accuracy to **95.7 %**. That contrast is the single
most useful thing I took from this project.

## Approach

```
train.csv + images/                     (10,031 labelled images)
        │
        ▼
NumPy loading (PIL, resize 224×224, uint8 0–255)
        │
        ▼
Stratified split 80 / 20        (8,024 train / 2,007 validation)
        │
        ▼
EfficientNetB0 (ImageNet, frozen)
        │
        ▼
GAP → BatchNorm → Dropout 0.5 → Dense(256) → BatchNorm → Dropout 0.5 → Dense(73)
        │
        ▼
Adam 1e-3 · 20 epochs · early stopping on val_accuracy (patience 10)
        │
        ▼
predict → submission.csv (ID, CLASS)
```

**Note on preprocessing.** Keras' EfficientNetB0 includes its own
`Rescaling(1/255)` and `Normalization` layers: it expects **raw 0–255 pixels**.
Dividing by 255 beforehand would rescale twice and break the model. The
`preprocess_input` function from `keras.applications.efficientnet` is a
pass-through for this architecture — it exists for API consistency, not because
it does anything.

## Challenges & Solutions

| Challenge | Symptom | Solution |
|---|---|---|
| **`Lambda` layer inside the model** | `ModelCheckpoint` failed — the model could not be saved, then could not be reloaded | Removed the `Lambda`; preprocessing moved out of the model into the data pipeline. A `Lambda` wraps a Python function, which Keras cannot serialise. |
| **Input scaling** | Model learned nothing on early attempts | EfficientNet already rescales internally, so the input must stay in **0–255**. No `/255`. |
| **Debugging the pipeline** | Could not learn even a single batch (~3 % accuracy) | Tested on one batch before launching a full run. A model that cannot overfit one batch points to the data pipeline, not the architecture. |

## Results

### Where the model fails

Errors are concentrated on visually adjacent classes:

| Class | Error rate | Likely confusion |
|---|---|---|
| frisbee | 23.1 % | disc sports at distance |
| figure_skating_pairs | 20.7 % | vs. men / women |
| luge | 20.0 % | vs. bobsled |
| snow_boarding | 17.4 % | vs. ski jumping |
| pommel_horse | 13.0 % | gymnastics apparatus |

Seven classes are classified perfectly (F1 = 1.00): curling, harness racing,
polo, rings, snowmobile racing, wheelchair basketball, wheelchair racing.

### The 3.9-point gap, and why it matters

Validation 95.67 % → private leaderboard 91.76 %.

**This gap is expected, and it is informative.** The validation set is a random
80/20 split *inside the training set*: images from the same event, venue or
photographer can land on both sides. The validation score is therefore
optimistic, and 0.9176 is the honest estimate of generalisation.

A grouped split (by event or by image source) would give a validation score
closer to the leaderboard. This is the first thing I would change.

The 1.2-point gap between public (0.9300) and private (0.9176) is, for its part,
ordinary sampling noise: the public split is roughly 550 images, so 1.2 points
is about seven images.

---

## What I learned

1. **Transfer learning beats from-scratch** by a factor of nearly four on this
   dataset (25 % → 96 % validation).
2. **A frozen backbone is not fine-tuning.** I initially described this project
   as fine-tuning; the code never unfroze the backbone. The distinction matters
   and I corrected it.
3. **Debug the pipeline on a single batch** before launching a full run. One
   batch that cannot be learned from points to the data pipeline, not the model.
4. **A `Lambda` layer is a trap.** It wraps a Python function, which Keras
   cannot serialise — so `ModelCheckpoint` fails and the model cannot be
   reloaded. Prefer moving preprocessing into the data pipeline, or use a
   serialisable `Rescaling` layer.
5. **Model-specific input conventions** matter more than they look: EfficientNet
   expects raw 0–255 because its rescaling is inside the model. Getting it wrong
   silently caps the accuracy.

---

## What I would improve next

1. **Enable the fine-tuning phase** (`RUN_FINE_TUNING = True`): unfreeze the
   last 100 layers, keep BatchNorm frozen, recompile at 1e-5. This is
   implemented in `src/model.py` but was not part of the reported run.
2. **Grouped validation split** to remove the same-event optimism described
   above.
3. **Class-balanced sampling** for the weakest classes (frisbee, pairs skating,
   luge).
4. **Test-time augmentation** on the 1,115 test images — cheap and usually
   worth a few tenths of a point.

---

## Repository structure

```
├── config.py             paths and hyperparameters
├── src/
│   ├── data.py           NumPy loading, stratified split
│   ├── model.py          EfficientNetB0 + head, fine-tuning helper
│   ├── train.py          training, callbacks, class_names export
│   ├── evaluate.py       per-class report, normalised confusion matrix
│   └── predict.py        batched inference, validated submission file
├── results/              metrics, training curves, confusion matrix
└── requirements.txt
```

## How to reproduce

```bash
git clone https://github.com/Nasreddine-EL-MAMMERI/Sport-images-classification.git
cd Sport-images-classification

python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Place the competition data under `data/`:

```
data/
├── train.csv
├── test.csv
└── sports_images/all_images/    # flat folder of images
```

Then:

```bash
python -m src.train       # writes models/best_model.keras + class_names.json
python -m src.evaluate    # writes results/classification_report.txt + figures
python -m src.predict     # writes submission.csv (ID, CLASS)
```

On Kaggle, `config.py` detects `/kaggle/input` automatically.

## Tech stack

Python 3.10+ · TensorFlow 2.18 / Keras · EfficientNetB0 (ImageNet) · NumPy ·
pandas · PIL · scikit-learn · Matplotlib · Seaborn

## Dataset

[Kaggle — OpenCV TF Project-2 Image Classification: Round 4](https://www.kaggle.com/competitions/opencv-tf-project-2-image-classification-round-4)

## License

MIT — see [LICENSE](LICENSE).
