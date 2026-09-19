'# Telco Customer Churn Prediction

## Objective
Identify customers likely to churn so a retention team can proactively engage them.

## Dataset observations
- Rows: 7,043
- Columns: 21
- Duplicate rows: 0
- Churn rate: 26.5%
- Train/test: 70% / 30%, stratified, random_state=42

## Feature engineering
1. `avg_monthly_charge` = TotalCharges / tenure, with MonthlyCharges used for tenure 0.
2. `service_count` = count of selected services with value Yes.

## Model comparison
                Model  Accuracy  Precision  Recall  F1 Score
  Tree A - controlled    0.7487     0.5183  0.7576    0.6155
      Tree B - deeper    0.7369     0.5028  0.7932    0.6155
Tree C - more complex    0.7198     0.4829  0.7790    0.5962

## Final model

The latest implementation uses the **tuned Decision Tree selected by GridSearchCV as the final model used by the API**.

GridSearchCV uses 5-fold cross-validation on the training set and selects the configuration using F1 score.

Latest selected parameters:
- `max_depth=6`
- `min_samples_leaf=50`
- `class_weight="balanced"`
- `random_state=42`

Best cross-validation F1 score: **0.6169**

The final model is obtained with:

```python
final_model = grid_search.best_estimator_
```

The held-out test set is used only for final evaluation, not for hyperparameter selection.

## Business interpretation
Recall is important in retention because false negatives are missed churners. Precision is also relevant because retention resources are finite. Threshold/operating policy should reflect the cost of each type of error.

## API

The project exposes a FastAPI service using the **fitted final Decision Tree pipeline**.

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then use `POST /predict` or Swagger at:

`http://127.0.0.1:8000/docs`

The API loads the serialized fitted pipeline from:

```text
model/churn_model.pkl
```

The saved pipeline contains feature engineering, preprocessing and the tuned Decision Tree. The API does not perform training or hyperparameter tuning.

## Structure
```text
customer_churn_project/
├── data/
├── notebook/churn_analysis.ipynb
├── model/churn_model.pkl
├── outputs/*.png
├── feature_engineering.py
├── app.py
├── business_layer.py
├── requirements.txt
├── README.md
└── sample_request.json
```

## Reusable feature engineering module

Feature engineering is implemented once in `feature_engineering.py` and imported by both the notebook and the API.

This avoids defining separate versions of the transformation logic and helps keep training-time and inference-time behavior consistent.

The module creates:
- `avg_monthly_charge`
- `service_count`

It also converts `TotalCharges` to numeric and removes `customerID` before prediction.

The notebook imports the function rather than defining a second copy:

```python
from feature_engineering import feature_engineering
```

When the fitted model is serialized, the pipeline therefore references the shared module rather than a notebook-local `__main__` function.

## Business Decision Layer

The solution now includes a separate business layer between ML prediction and retention execution:

```text
Customer Data
     ↓
Preparation + Feature Engineering
     ↓
Churn Model
     ↓
Churn Probability
     ↓
Risk Segmentation
     ↓
Retention Action
     ↓
Outcome Tracking
     ↓
Model / Strategy Improvement
```

### Risk bands

| Risk | Prototype rule | Action |
|---|---|---|
| High Risk | >= 0.75 | Prioritized retention outreach |
| Medium Risk | 0.50–0.7499 | Proactive digital engagement and targeted retention |
| Low Risk | < 0.50 | Normal lifecycle engagement |

These thresholds are operational prototype rules. They should be calibrated in production using intervention cost, customer value and observed retention outcomes.

### Prioritization

A simple prototype `priority_score` is calculated as:

`churn_probability × MonthlyCharges`

This helps surface customers where both churn risk and current monthly value are material. It is a prioritization heuristic, **not a causal estimate of revenue that will be retained**.

### Closed-loop retention

For every intervention, capture the risk score, intervention type, incentive cost, response, retention outcome and revenue/profit impact. This allows the organization to evaluate whether interventions actually reduce churn rather than only measuring model accuracy.

### API response

The API retains the required prediction and probability fields and additionally returns the business decision:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82,
  "risk_band": "High Risk",
  "recommended_action": "Prioritized retention outreach"
}
```

## Model serialization and API verification

After GridSearchCV completes, the fitted final model is obtained from:

```python
final_model = grid_search.best_estimator_
```

Before saving, the model is tested with `predict()` and `predict_proba()`.

The fitted pipeline is then serialized:

```python
with open("../model/churn_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
```

The saved artifact is loaded again and tested before starting Uvicorn. This verifies that the API receives a fitted model and that the shared `feature_engineering` module can be resolved when the pickle is loaded outside the notebook.

## Bonus enhancements implemented

- **Class imbalance handling:** `class_weight="balanced"` is used for the Decision Tree and Logistic Regression.
- **Hyperparameter tuning:** 5-fold `GridSearchCV` is performed only on the training set using F1 score; the held-out test set is not used during tuning.
- **Additional model comparison:** Logistic Regression uses the same feature-engineering and preprocessing pipeline as the Decision Tree.
- **Additional EDA:** Churn Rate by Tenure Group has been added, bringing the project to **7 meaningful EDA visualizations**.
- **Saved bonus artifacts:** `model/tuned_decision_tree.pkl`, `model/logistic_regression.pkl`, and `outputs/bonus_model_comparison.json/.csv`.

The **tuned Decision Tree is the primary API model**. It satisfies the Decision Tree requirement while using GridSearchCV to select its hyperparameters. The complete fitted pipeline is saved as `model/churn_model.pkl`.
