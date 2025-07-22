# Fraud Detection Cases for E-Commerce and Bank Transactions
![Black Formatting](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort Imports](https://img.shields.io/badge/imports-isort-blue.svg)
![Flake8 Lint](https://img.shields.io/badge/lint-flake8-yellow.svg)
[![CI](https://github.com/nuhaminae/Fraud-Detection-Cases-for-E-Commerce-and-Bank-Transactions/actions/workflows/CI.yml/badge.svg)](https://github.com/nuhaminae/Fraud-Detection-Cases-for-E-Commerce-and-Bank-Transactions/actions/workflows/CI.yml)

## Overview
This project tackles the detection of fraudulent activities across e-commerce and banking transactions using machine learning and geolocation intelligence. It emphasises strong data preprocessing pipelines, class imbalance handling, and interpretable modelling to balance security and customer experience.

---
## Key Features
- Modular preprocessing scripts for fraud detection datasets.
- Feature engineering tailored to time-based patterns and transaction velocity.
- Class imbalance handling using SMOTE with strict datatype safeguards.
- Scaled, reproducible train/test splits for both bank and e-commerce transactions.
- CI pipelines and pre-commit checks for code consistency and quality.
- SHAP-based model explainability integrated for transparency. (next step)

---
## Table of Contents
- [Project Background](#project-background)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contribution](#contribution)
- [Project Highlights](#project-highlights)
- [Project Status](#project-status)

---
## Project Background
Fraud detection is a high-stakes domain where false positives disrupt user trust and false negatives cause financial losses. This project simulates real-world detection scenarios by preprocessing, modelling, and interpreting two datasets: e-commerce transactions and anonymised bank credit records.

It draws on business challenges faced by Adey Innovations Inc., where detection models must be explainable, accurate, and scalable for operational deployment

---
## Data Sources
1. _**CreditCard.csv**_
    * Bank transaction data with PCA-transformed features (V1–V28) and fraud labels.
    * Challenge: Extremely imbalanced and anonymised features.
2. _**FraudData.csv**_
    * Transaction metadata for e-commerce purchases.
    * Includes user and device identifiers, timestamps, purchase values, IP addresses, and fraud labels.
    * Challenge: High class imbalance + potential data leakage from IDs.
3. _**IpAddressToCountry.csv**_
    * Mapping IP ranges to country for geolocation enrichment.
    * Integrated via integer casting and merge logic.

---
## Project Structure
```
fraud_detection_project/
├── dvc/                               # Data Version Control
├── .github/                           # CI workflows
├── data/
│   ├── raw/                           # Original datasets
│   └── processed/                     # Cleaned and transformed datasets
├── insights/                          # Plots and charts for reporting
├── notebooks/                         # Exploratory Data Analysis and Feature Engineering notebooks
│   ├── 01_eda.ipynb                   
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb              (next step)
│   └── 04_model_explainability.ipynb   (next step)
├── scripts/                           # Core scripts
│   ├── _01_data_preprocessing.py
│   ├── _02_feature_engineering.py
│   ├── _03_train_model.py              (next step)
│   └── _04_explain_model.py            (next step)
├── tests/
├── .dvcignore
├── .flake8
├── .gitignore                         # Ignore unnecessary files
├── .pre-commit-config.yaml            # Pre-commit configuration
├── format.ps1                         # Formatting
├── pyproject.toml
├── requirements.txt                   # Pip install fallback
├── README.md                          # Project overview and setup instructions
└── requirements.txt                   # Pip install fallback
```

---
## Installation
### Prerequisites

- Python 3.8 or newer (Python 3.12 recommended)
- `pip` (Python package manager)
- [DVC](https://dvc.org/) (for data version control)
- [Git](https://git-scm.com/)

### Setup
```bash
# Clone repo
git clone https://github.com/nuhaminae/Fraud-Detection-Cases-for-E-Commerce-and-Bank-Transactions
cd https://github.com/nuhaminae/Fraud-Detection-Cases-for-E-Commerce-and-Bank-Transactions
____________________________________________
# Create and activate virtual environment
python -m venv .fraudvenv
.fraudvenv\Scripts\activate      # On Windows
source .fraudvenv/bin/activate   # On Unix/macOS
____________________________________________
# Install dependencies
pip install -r requirements.txt
____________________________________________
# Install and activate pre-commit hooks
pip install pre-commit
pre-commit install
____________________________________________
# (Optional) Pull DVC data
dvc pull
```

---
## Usage
### Code Quality

This project uses pre-commit hooks to automatically format and lint `.py` and `.ipynb` files using:

|Tool	      | Purpose                                       |
|:---------:|-----------------------------------------------|
| Black	    |Enforces consistent code formatting            | 
| isort	    |Sorts and organises import statements          |
| Flake8		|Lints Python code for style issues             |
| nbQA		  |Runs Black, isort, and Flake8 inside notebooks |


---
## Contribution
Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss what you would like to change.
Make sure to follow best practices for version control, testing, and documentation.

---
## Project Highlights
- Robust class-aware preprocessing and type casting to preserve fraud signals.
- Transaction frequency and velocity features per user/device.
- Engineered temporal features like Hour_Of_Day, Day_Of_Week, Time_Since_Signup.
- Advanced class imbalance correction using SMOTE post-encoding and datatype filtering.
- Defensive coding practices like safe_relpath, clean feature selection, and testable pipelines.
- Visual insights powered by SHAP, enhancing model explainability and business trust. (Next steps) 

---
## Project Status
Project is still underway. Checkout the commit history [here](https://github.com/nuhaminae/Fraud-Detection-Cases-for-E-Commerce-and-Bank-Transactions). 
