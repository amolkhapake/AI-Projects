"""
Project 1: ANN - Banking Loan Default Prediction
--------------------------------------------------
pip install tensorflow scikit-learn pandas numpy imbalanced-learn shap

Predicts probability of loan default from applicant financial features.
Uses synthetic data structured like a real credit dataset - swap
`load_data()` with a read from your bank's data warehouse/feature store.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras import layers


def load_data(n=20000, seed=42):
    """Synthetic stand-in for a real loan applicant dataset."""
    rng = np.random.default_rng(seed)
    income = rng.normal(60000, 20000, n).clip(15000, 300000)
    credit_score = rng.normal(650, 80, n).clip(300, 850)
    debt_to_income = rng.uniform(0, 0.8, n)
    loan_amount = rng.normal(20000, 8000, n).clip(1000, 100000)
    employment_years = rng.integers(0, 30, n)
    num_late_payments = rng.poisson(1.5, n)

    # default probability driven by a realistic (nonlinear) combination of features
    risk_score = (
        -0.00003 * income
        - 0.01 * credit_score
        + 3.0 * debt_to_income
        + 0.00004 * loan_amount
        - 0.03 * employment_years
        + 0.25 * num_late_payments
    )
    prob_default = 1 / (1 + np.exp(-(risk_score + 4)))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "debt_to_income": debt_to_income,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "num_late_payments": num_late_payments,
        "default": default,
    })
    return df


def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    return model


def main():
    df = load_data()
    X = df.drop(columns="default")
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle severe class imbalance (defaults are rare in real data)
    print(f"Default rate before SMOTE: {y_train.mean():.3f}")
    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train)
    print(f"Default rate after SMOTE:  {y_train_bal.mean():.3f}")

    model = build_model(X_train_bal.shape[1])
    model.fit(
        X_train_bal, y_train_bal,
        validation_split=0.1,
        epochs=30,
        batch_size=64,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2,
    )

    y_pred_proba = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\n--- Test Set Performance ---")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

    # --- Explainability (required for banking/regulatory use) ---
    try:
        import shap
        explainer = shap.Explainer(model, X_train_scaled[:100])
        shap_values = explainer(X_test_scaled[:50])
        print("\nSHAP explainability computed for first 50 test applicants.")
        print("Mean |SHAP value| per feature (higher = more influence on decision):")
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        for col, val in sorted(zip(X.columns, mean_abs), key=lambda x: -x[1]):
            print(f"  {col:20s}: {val:.4f}")
    except Exception as e:
        print(f"SHAP step skipped ({e}). Install `shap` to enable explainability.")

    model.save("loan_default_ann.keras")
    print("\nModel saved to loan_default_ann.keras")


if __name__ == "__main__":
    main()
