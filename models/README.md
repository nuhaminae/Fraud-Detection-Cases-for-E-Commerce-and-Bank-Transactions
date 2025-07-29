## Model Overview
This directory contains model training, evaluation, and selection logic for both **Credit** and **Fraud** transaction datasets. Models are trained using imbalanced classification techniques and evaluated with precision-focused metrics to optimize fraud detection sensitivity.

> Model artifacts are **version-controlled using DVC** and saved as `.pkl` files for reuse, enabling reproducible experimentation and deployment.

### Architectures Used
| Dataset | Baseline            | Random Forest   | Gradient Boosting  | XGBoost |
|---------|---------------------|-----------------|--------------------|---------|
| Credit  | Logistic Regression | ✔️              | ✔️                | ✔️      |
| Fraud   | Logistic Regression | ✔️              | ✔️                | ✔️      |

Each model is trained on **preprocessed data with engineered features**, including time-based patterns and geolocation signals.

---
## Evaluation Strategy
Models are evaluated using metrics tailored to imbalanced classification, capturing real-world impact:

- **F1 Score**: Balances precision and recall to reduce both false positives and false negatives.
- **AUC-PR**: Preferred over ROC-AUC due to better reflection of model behavior on highly imbalanced datasets.
- **Confusion Matrix**: Essential for monitoring trade-offs between customer friction and security.

Visualization includes precision-recall curves, ROC curves, and confusion matrices.

---
## Best Model Selection
For each dataset, the model with the **highest F1 Score** is selected and saved using `joblib`. Naming convention follows:

```
{label}_best_model_{model_name}.pkl
```

This ensures clear identification and reuse within modular pipelines.

---
## Retraining Instructions
To retrain models:

1. Replace datasets in the `data/` folder (raw or processed).
2. Run the `train_and_evaluate()` method in the modeling script.
3. Modular **feature engineering pipelines** are auto-triggered before training.
4. Best model is saved in `models/` and evaluation plots are refreshed.

> No need to modify evaluation logic unless new models or metrics are added.

---
## Extras
- Feature engineering supports time-based and velocity signals across users/devices.
- SHAP workflows (summary & force plots) under `insights/explainer/` reveal **global and local model interpretability**.
- All diagnostic plots are saved in reproducible formats (`.png`) to support reporting and version tracking.