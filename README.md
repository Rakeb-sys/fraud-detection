# Detection of Fraud Cases for E-commerce and Bank Transactions

An end-to-end machine learning project developed for **Adey Innovations Inc.** to detect and reduce fraudulent activities across two distinct transaction streams: Context-rich E-commerce transactions and PCA-anonymized Bank Credit Card transactions.

---

## 📌 Project Overview

Effective fraud detection balances two competing operational costs:

* **False Positives:** Flagging legitimate transactions, which frustrates customers and erodes brand trust.
* **False Negatives:** Missing actual fraud, leading directly to immediate financial loss.

This repository implements parallel machine learning pipelines tailored to handle two vastly different data shapes under severe class imbalance, utilizing **SMOTE/undersampling**, advanced tree-based ensembles (**XGBoost/LightGBM**), and **SHAP** for regulatory explainability.

---

## 📁 Repository Structure

```text
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml       # CI pipeline executing automated testing
├── data/                       # Local data storage (Git ignored)
│   ├── raw/                    # Original unchanged datasets
│   └── processed/              # Cleaned, geolocated, and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb    # Profiling & pattern exploration for E-commerce data
│   ├── eda-creditcard.ipynb   # Profiling & distribution exploration for Bank data
│   ├── feature-engineering.ipynb # Geolocation lookup, velocity, and temporal features
│   ├── modeling.ipynb          # Baseline and ensemble training with resampling
│   ├── shap-explainability.ipynb # Global and local explainability visualizations
│   └── README.md
├── src/                        # Modular source code for production deployment
│   ├── __init__.py
│   ├── data_preprocessing.py   # Cleaning and IP integer conversion utils
│   ├── feature_engineering.py  # Feature creation and geolocation merger
│   ├── model_training.py       # Training, cross-validation, and metrics tracking
│   └── explainability.py       # SHAP plotting abstractions
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py        # PyTest unit tests for data transformations
├── models/                     # Serialized model artifacts (.pkl / .joblib)
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt            # Project environment dependencies
├── README.md                   # Project documentation (This file)
└── .gitignore

```

---

## 📊 Dataset Reference

The system processes three primary datasets across two independent channels:

### 1. E-Commerce Pipeline

* **`Fraud_Data.csv`**: Contains explicit device, behavioral, and contextual fields (`user_id`, `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`, `class`).
* **`IpAddress_to_Country.csv`**: Maps numeric IP ranges (`lower_bound_ip_address`, `upper_bound_ip_address`) to their respective `country` code.

### 2. Bank Credit Pipeline

* **`creditcard.csv`**: Contains anonymized bank operations features derived via Principal Component Analysis (`Time`, `V1–V28`, `Amount`, `Class`).

> ⚠️ **Note:** Both target datasets are highly imbalanced, where fraudulent activities comprise a small fraction of total entries. Overall metrics like **Accuracy** are discarded in favor of **AUC-PR** and **F1-Score**.

---

## 🛠️ Installation & Setup

### Prerequisites

* Python 3.10 or higher
* `pip` package manager

### Environment Configuration

1. **Clone the repository:**
```bash
git clone https://github.com/rakeb-sys/fraud-detection.git
cd fraud-detection

```


2. **Set up a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt

```


4. **Populate Data Storage:**
Place the downloaded raw CSV files into the `data/raw/` directory before running any modules or notebooks.

---

## 🚀 Execution Workflow

### Step 1: Preprocessing & Feature Engineering

Run data transformation steps, including range-based IP-to-country lookup and behavioral calculations (e.g., `time_since_signup`, `purchase_velocity`).

```bash
python scripts/run_preprocessing.py

```

### Step 2: Model Training & Evaluation

Train the baseline models alongside advanced tree-based ensembles (XGBoost/LightGBM) using stratified 5-fold cross-validation.

```bash
python scripts/run_modeling.py

```

### Step 3: Run Unit Tests

Ensure system reliability and check logic integrity with `pytest`:

```bash
pytest tests/

```

---

## 📈 Methodology Summary

### Pipeline Prep & Engineering

* **IP-to-Country Conversion:** Formats string IP addresses to integers to perform range-based merging via `pandas.merge_asof`.
* **Behavioral Analysis:** Derives time deltas (`purchase_time` $-$ `signup_time`) to capture immediate-purchase bot networks, along with transactional frequency over sliding windows.
* **Resampling:** Applies **SMOTE** strictly within the training folds to avoid data leakage into validation distributions.

### Modeling and Strategy

* **Baseline Configuration:** Implements an interpretable L1/L2 penalized Logistic Regression.
* **Ensemble Optimization:** Fine-tunes high-performance gradient boosting models (XGBoost, LightGBM) to maximize the **Precision-Recall Area Under Curve (AUC-PR)**.

### Explainability (XAI)

* **SHAP (SHapley Additive exPlanations):** Computes global feature impact configurations and builds local prediction force-plots highlighting true positives, false positives, and false negatives to meet strict fintech compliance criteria.

