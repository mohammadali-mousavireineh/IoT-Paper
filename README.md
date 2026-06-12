# IoT-Paper: Machine Learning for IoT / Windows Dataset Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/ML-LightGBM%20%7C%20Scikit--learn-green" />
  <img src="https://img.shields.io/badge/Notebook-Jupyter-orange" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-purple" />
</p>

A clean, reproducible machine-learning project for experimenting with IoT/Windows dataset classification.  
The original repository contained a Jupyter notebook and a CSV dataset. This upgraded version reorganizes the work into a GitHub-ready research project with a professional README, source code modules, configuration files, reproducible scripts, and documentation.

> **Research note:** The original notebook reported very high scores with models such as LightGBM and Decision Tree. This version adds a safer evaluation pipeline to reduce the risk of data leakage and overfitting.

---

## ✨ Features

- Clean project structure for GitHub and research presentation
- Reproducible ML pipeline using `scikit-learn` and `lightgbm`
- Binary and multiclass classification support
- Data validation, preprocessing, model training, evaluation, and report generation
- Confusion matrix and classification report outputs
- Configuration-based workflow
- GitHub Actions workflow for basic Python checks
- Ready-to-customize notebook folder

---

## 📁 Project Structure

```text
IoT-Paper-Professional/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/              # Put windows7_dataset.csv here
│   └── processed/        # Generated clean datasets
├── notebooks/
│   └── README.md
├── reports/
│   └── figures/          # Saved plots and evaluation figures
├── src/
│   └── iot_paper/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── train.py
│       └── evaluate.py
├── .github/workflows/
│   └── python-checks.yml
├── .gitignore
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/mohammadali-mousavireineh/IoT-Paper.git
cd IoT-Paper
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Put the dataset in the correct folder

Place your dataset here:

```text
data/raw/windows7_dataset.csv
```

### 5. Run training

```bash
python -m iot_paper.train --config configs/default.yaml
```

---

## 🧠 Machine Learning Pipeline

The pipeline follows these steps:

1. Load raw CSV data
2. Detect and remove empty / duplicated columns
3. Split features and target label
4. Encode categorical values
5. Train a baseline model
6. Evaluate with accuracy, F1-score, precision, recall, and confusion matrix
7. Save results under `reports/`

---

## ⚠️ Important Research Warning: Overfitting & Leakage

Very high accuracy, such as 99%, can happen for two very different reasons:

1. **The model is genuinely good**, or
2. **The dataset leaks the answer** through duplicated rows, target-like columns, timestamps, IDs, or train/test contamination.

Before claiming strong performance, check:

- Are there duplicated rows across train/test sets?
- Are there columns that directly reveal the target?
- Is the split random when it should be time-based or device-based?
- Are IDs, session labels, or post-event columns included as features?
- Does cross-validation show similar performance?

---

## 📊 Suggested Experiments

- Decision Tree baseline
- Random Forest baseline
- LightGBM multiclass classifier
- Binary classification version
- Feature importance analysis
- Ablation study: remove suspicious columns and retrain
- Stratified K-fold cross-validation
- Time-aware or group-aware split if session/device IDs exist

---

## 🧪 Example Commands

Train using the default configuration:

```bash
python -m iot_paper.train --config configs/default.yaml
```

Run evaluation only after saving predictions:

```bash
python -m iot_paper.evaluate --predictions reports/predictions.csv
```

---

## 📌 Dataset

The dataset is not redistributed in this professional template. Put the original CSV file in:

```text
data/raw/windows7_dataset.csv
```

The original GitHub repository includes `windows7_dataset.csv` and `Untitled-1.ipynb`.

---

## 🛣️ Roadmap

- [ ] Clean and rename the original notebook
- [ ] Add exploratory data analysis notebook
- [ ] Add leakage detection report
- [ ] Add model comparison table
- [ ] Add feature importance visualizations
- [ ] Add paper-style result summary
- [ ] Add Dockerfile
- [ ] Add Streamlit demo dashboard

---

## 👤 Author

**Mohammadali Mousavireineh**  
GitHub: [@mohammadali-mousavireineh](https://github.com/mohammadali-mousavireineh)

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
