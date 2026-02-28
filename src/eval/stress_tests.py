import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

MODEL_PATH = "artifacts/model_baseline.joblib"
DATA_PATH = "data/credit_default.xls"


# ---------------------------------------------------
# Feature Perturbation (Robustness Testing)
# ---------------------------------------------------
def perturb_features(X, epsilon=0.02):
    X_perturbed = X.copy()

    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.number):
            std = X[col].std()
            noise = np.random.normal(0, epsilon * std, size=len(X))
            X_perturbed[col] += noise

    return X_perturbed


# ---------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------
def expected_calibration_error(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(probs, bins) - 1

    ece = 0.0

    for i in range(n_bins):
        mask = binids == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = probs[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.mean()

    return ece


# ---------------------------------------------------
# Main Evaluation
# ---------------------------------------------------
def main():
    df = pd.read_excel(DATA_PATH, header=1)
    target_col = "default payment next month"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    model = joblib.load(MODEL_PATH)

    clean_probs = model.predict_proba(X)[:, 1]
    clean_preds = (clean_probs > 0.5).astype(int)

    clean_auc = roc_auc_score(y, clean_probs)

    high_conf_wrong_clean = (
        (clean_preds != y) & (clean_probs >= 0.9)
    ).sum()

    print("\n===== CLEAN PERFORMANCE =====")
    print(f"ROC-AUC: {clean_auc:.4f}")
    print(f"High-confidence wrong (>=0.9): {high_conf_wrong_clean}")

    # ---------------------------------------------------
    # Calibration
    # ---------------------------------------------------
    ece = expected_calibration_error(y, clean_probs)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # ---------------------------------------------------
    # Abstention (Reject Option)
    # ---------------------------------------------------
    lower, upper = 0.4, 0.6
    abstain_mask = (clean_probs > lower) & (clean_probs < upper)

    accepted_mask = ~abstain_mask
    coverage = accepted_mask.mean()

    accepted_preds = clean_preds[accepted_mask]
    accepted_y = y[accepted_mask]
    accepted_probs = clean_probs[accepted_mask]

    accuracy = (accepted_preds == accepted_y).mean()

    high_conf_wrong_accepted = (
        (accepted_preds != accepted_y) & (accepted_probs >= 0.9)
    ).sum()

    print("\n===== ABSTENTION RESULTS =====")
    print(f"Coverage: {coverage:.3f}")
    print(f"Accuracy (accepted only): {accuracy:.3f}")
    print(f"High-confidence wrong (accepted): {high_conf_wrong_accepted}")

    # ---------------------------------------------------
    # Robustness Tests
    # ---------------------------------------------------
    print("\n===== ROBUSTNESS TESTS =====")

    for eps in [0.01, 0.02, 0.05, 0.1]:
        X_perturbed = perturb_features(X, epsilon=eps)

        probs = model.predict_proba(X_perturbed)[:, 1]
        preds = (probs > 0.5).astype(int)

        auc = roc_auc_score(y, probs)
        drop = clean_auc - auc

        high_conf_wrong = (
            (preds != y) & (probs >= 0.9)
        ).sum()

        print(f"\n--- Epsilon {eps} ---")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"AUC Drop: {drop:.4f}")
        print(f"High-confidence wrong: {high_conf_wrong}")


if __name__ == "__main__":
    main()