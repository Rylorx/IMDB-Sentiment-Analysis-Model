# IMDB Sentiment Analysis Model

## 📌 Project Overview
This repository contains a sentiment analysis project trained on the IMDb 50,000-review dataset (25k positive, 25k negative). The model classifies whether a movie review is **positive** or **negative** using deep learning (Keras/TensorFlow). A saved tokenizer (`tokenizer.pkl`) and trained model (`IMDB_model.h5`) are used for inference.

## 📁 Dataset
- Dataset: **IMDb Movie Review Dataset (50,000 reviews)**
- Public sources:
  - Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  - Stanford: http://ai.stanford.edu/~amaas/data/sentiment/
- Format: CSV containing `review` and `sentiment` columns

✅ Example row from dataset:

```
"One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked...",positive
```

## 🧠 Project Structure
```
project-root/
├── data/                     # raw dataset (ignored in Git)
├── models/                   # saved model + tokenizer (ignored)
│   ├── IMDB_model.h5
│   └── tokenizer.pkl
├── src/                      # preprocessing + training scripts
├── notebooks/                # optional Jupyter notebooks
├── README.md
├── requirements.txt
└── .gitignore
```

## ⚙️ Model Pipeline
1. Load and preprocess text (clean HTML, lowercase, remove punctuation, etc.)
2. Tokenize and pad sequences (`tokenizer.pkl`)
3. Train a neural network (LSTM/CNN/etc.) on padded sequences
4. Save trained model (`IMDB_model.h5`) and tokenizer for later inference
5. Load model + tokenizer to classify new reviews

## ❗ Why Model & Dataset Files Aren’t in Git
Large binary artifacts (datasets, `.h5` models, `.pkl` tokenizers) are not committed to Git because:
- They quickly bloat repo size
- GitHub hard-limits single files >100MB
- They update frequently and do not diff well

Instead, they should be stored using:
- Git LFS
- HuggingFace Hub
- Google Drive / S3 / GitHub Releases
- Or a download script (`download_data.py`)

## 📋 Requirements
- Python 3.x
- numpy
- pandas
- tensorflow / keras
- scikit-learn
- nltk (or similar for text cleaning)

Install using:
```
pip install -r requirements.txt
```

## ▶️ Usage

### 1. Clone repository
```
git clone https://github.com/<your-username>/IMDB-Sentiment-Analysis-Model.git
cd IMDB-Sentiment-Analysis-Model
```

### 2. Create virtual environment
macOS/Linux:
```
python -m venv .venv
source .venv/bin/activate
```

Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download dataset
- Place CSV file into `data/` OR use a download script if provided.

### 5. Train or run inference
```
python src/train.py      # train model
python src/predict.py    # classify new text
```

## ✅ Evaluation
Add accuracy, loss curves, confusion matrix, or sample predictions here.

## 📜 Acknowledgements
- Dataset: “Large Movie Review Dataset” — Andrew Maas et al. (Stanford AI Lab, 2011)
- Code: Created by <your name>
- Libraries: TensorFlow, Keras, NumPy, Pandas, Scikit-Learn

## 🤝 Contributing
Pull requests are welcome — feel free to open issues or suggest improvements.

⭐ If you found this useful, consider giving the repo a star!
