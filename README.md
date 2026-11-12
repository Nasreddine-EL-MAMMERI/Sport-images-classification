# 🏆 Sport Image Classification — Fine-Tuning EfficientNetB0

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-GPU-20BEFF)
![Public Leaderboard](https://img.shields.io/badge/Public%20LB-0.92998-green)

> **73 sports classes** — 10,031 training images — **93.0% on Kaggle Public Leaderboard**  
> Project 2: Kaggle Competition — Image Classification (100 points)

---

## 📋 Table of Contents

1. [The Problem](#the-problem)
2. [The Approach](#the-approach)
3. [Challenges & Solutions](#challenges--solutions)
4. [Results](#results)
5. [What I Learned](#what-i-learned)
6. [How to Reproduce](#how-to-reproduce)
7. [Tech Stack](#tech-stack)
8. [License](#license)

---

## 🎯 The Problem

**Goal:** Classify sports images into **73 different classes** (air hockey, basketball, curling, figure skating, etc.) with at least **86% accuracy** on the Kaggle Public Test Leaderboard.

**Constraints:**
- ⏱️ Limited training time (local CPU too slow)
- 📊 Imbalanced dataset (73 classes, ~140 images per class on average)
- 🖼️ Images of variable sizes
- 🚫 Notebook must **not** be published on the Kaggle competition page (lab rule)

---

## 🧠 The Approach

### Why EfficientNetB0?

I first trained a **CNN from scratch**. Result: ~25% accuracy, unstable. With 73 classes and only ~10,000 images, a from-scratch model cannot learn rich enough visual features.

I switched to **Transfer Learning** with **EfficientNetB0** pre-trained on ImageNet. The backbone extracts relevant visual features (shapes, textures, edges), and I train only a custom classification head for my 73 classes.

### Training Pipeline

Data (CSV + image folders)
↓
NumPy loading (PIL, resize 224×224, uint8 0-255)
↓
Stratified split 80% train / 20% validation
↓
EfficientNet preprocessing (preprocess_input)
↓
Model: EfficientNetB0 (frozen) + GlobalAvgPooling + BatchNorm + Dropout + Dense(73)
↓
Phase 1: Train head only (Adam, lr=1e-3)
↓
Save best model (val_accuracy)
↓
Predict on test → submission.csv

---

## ⚠️ Challenges & Solutions

| Challenge | Symptom | Solution |
|-----------|---------|----------|
| **tf.data pipeline** | Model couldn't learn even on 1 batch (~3% accuracy) | Switched to **direct NumPy loading** with PIL — more reliable and easier to debug |
| **Lambda layer** | Impossible to reload saved model (`NotImplementedError`) | Removed the `Lambda` layer; preprocessing done upstream on NumPy arrays |
| **Kaggle GPU** | Local CPU too slow (hours per epoch) | Trained on **Kaggle GPU Tesla T4** — minutes per epoch |
| **GitHub storage** | Files > 100 MB rejected (data, models) | Strict `.gitignore`: `data/`, `models/`, `*.keras` excluded |

---

## 📊 Results

### Key Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Validation Accuracy** | **95.67%** | — | ✅ |
| **Public Leaderboard** | **0.92998 (93.0%)** | ≥ 86% (min points) | ✅ |
| **Public Leaderboard** | **0.92998 (93.0%)** | ≥ 92% (max points) | ✅ **70/70** |
| Classes predicted | 73/73 | — | ✅ |

### Performance by Class

The model achieves an average **precision of 0.96** and **recall of 0.96** on validation. Some classes are perfectly recognized (e.g., *curling*, *polo*, *rings*, *snowmobile_racing*, *wheelchair_basketball* with F1-score of 1.00), while others are harder (e.g., *figure_skating_pairs*, *frisbee*, *giant_slalom* around 0.85–0.89).

### Visualizations

#### Training Curves

![Training curves](results/training_curves.png)

*Validation accuracy rises quickly and stabilizes around 95% without significant overfitting (reasonable train/validation gap).*

#### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

*The dominant diagonal confirms the model correctly classifies most images. Errors concentrate on visually similar classes (e.g., figure skating men/women/pairs).*

---

## 💡 What I Learned

1. **Transfer Learning > From Scratch** for small multi-class datasets.
2. **NumPy > tf.data** when the data pipeline becomes a black box hard to debug.
3. **Model-specific preprocessing** matters (EfficientNet expects `preprocess_input`, not just `/255`).
4. **Kaggle GPU** is essential for fast iteration on computer vision models.
5. **A good README with storytelling** communicates your work as well as the code does.

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/Nasreddine-EL-MAMMERI/Sport-images-classification.git
cd Sport-images-classification

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\activate        # Windows
# source .venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data manually from Kaggle
# Place images in data/Train/, data/Valid/, data/Test/
# and CSV files in data/

# 5. Training (recommended on Kaggle GPU)
# Import the notebook in Kaggle, enable GPU accelerator, run all cells

🛠️ Tech Stack
Component	Technology
Language	Python 3.10+
Framework	TensorFlow / Keras 2.x
Model	EfficientNetB0 (Transfer Learning)
Data	NumPy, Pandas, PIL
Visualization	Matplotlib, Seaborn
Training	Kaggle GPU (Tesla T4)
Version Control	Git + GitHub
📄 License
MIT License — see 
LICENSE
 for details.

🙏 Acknowledgements
Dataset & competition: Kaggle — OpenCV TF Project 2
Framework: TensorFlow / Keras