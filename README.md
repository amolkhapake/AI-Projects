# Loan Default Risk Predictor (ANN) — Banking

Predicts the probability that a loan applicant will default, using an
artificial neural network trained on applicant financial features.

## Problem
Banks need to price risk and decide approvals quickly. This model gives
a default probability score to support (not replace) underwriting.

## Architecture
- Input: 6 features (income, credit score, debt-to-income, loan amount,
  employment years, late payment history)
- ANN: Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dense(1, sigmoid)
- Trained with class-imbalance handling (SMOTE) since defaults are rare
- Explainability via SHAP for regulatory compliance

## Files
- `train.py` — data loading, model architecture, training, evaluation, SHAP
- `app.py` — Streamlit UI: enter applicant details, get a risk score
- `api.py` — FastAPI service exposing `/predict` for system integration

## Run it

```bash
pip install tensorflow scikit-learn pandas numpy imbalanced-learn shap streamlit fastapi uvicorn

# Train + evaluate from the command line
python train.py

# Interactive UI
streamlit run app.py

# REST API
uvicorn api:app --reload --port 8000
# then POST to http://localhost:8000/predict, or view docs at /docs
```

### Example API call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 55000, "credit_score": 650, "debt_to_income": 0.3,
       "loan_amount": 20000, "employment_years": 5, "num_late_payments": 0}'
```

## Interview talking points
- Why ANN over logistic regression: captures non-linear interactions between
  debt ratio, credit score, and payment history that a linear model misses.
- Why SMOTE: defaults are a small minority class; without rebalancing, the
  model would just predict "no default" for everyone and still look accurate.
- Why SHAP: banking regulators (and applicants) are entitled to know why a
  decision was made — a black-box score alone isn't defensible.
- Production gap addressed: model connects to a feature store so training
  and real-time scoring use identical feature computation logic, avoiding
  train/serve skew.

## What's synthetic vs. real here
Data is synthetically generated (`load_data()` in `train.py`) to mimic a
real credit dataset's structure and correlations. Swap in your bank's data
warehouse extract (or a public dataset like Lending Club) for real training.
