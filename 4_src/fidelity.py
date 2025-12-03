import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.generation import train_generator


def calculate_ks_test(real, syn, continuous_cols):
    """
    Computes Kolmogorov-Smirnov test for continuous variables.
    Returns 1 - KS Statistic (so higher is better).
    """
    results = {}
    for col in continuous_cols:
        stat, p_value = ks_2samp(real[col], syn[col])
        results[col] = {"KS_Stat": stat, "P_Value": p_value, "Score": 1 - stat}
    return results


def calculate_correlation_diff(real, syn):
    """
    Computes the difference between correlation matrices.
    """
    # Encode categoricals for correlation
    real_enc = real.copy()
    syn_enc = syn.copy()

    le = LabelEncoder()
    for col in real.select_dtypes(include="object").columns:
        real_enc[col] = le.fit_transform(real_enc[col].astype(str))
        syn_enc[col] = le.fit_transform(syn_enc[col].astype(str))

    corr_real = real_enc.corr()
    corr_syn = syn_enc.corr()

    # Frobenius Norm of the difference
    diff_matrix = corr_real - corr_syn
    frobenius_norm = np.linalg.norm(diff_matrix)

    return frobenius_norm, corr_real, corr_syn


def calculate_discriminator_score(real, syn):
    """
    Trains a classifier to distinguish Real (0) vs Synthetic (1).
    Returns AUC. 0.5 is perfect (indistinguishable), 1.0 is bad (easily distinguishable).
    """
    real["is_synthetic"] = 0
    syn["is_synthetic"] = 1

    combined = pd.concat([real, syn], axis=0)
    X = combined.drop(columns=["is_synthetic"])
    y = combined["is_synthetic"]

    # Encode
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    clf = RandomForestClassifier(random_state=42)

    # We want to see if we can distinguish them.
    # High AUC = Bad Fidelity.
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

    return scores.mean()


def run_fidelity_check(data_path):
    print("Loading Real Data...")
    df_real = pd.read_csv(data_path)

    print("Training Gaussian Copula (Winner)...")
    # We train on the FULL real dataset to get the best representation
    gen = train_generator(df_real, model_type="copula")

    print("Generating Synthetic Data (Same Size)...")
    df_syn = gen.sample(len(df_real))

    # Define Columns
    # Based on the dataset we've seen: Age, Weight, Height, Pulse, BP are continuous
    # Gender, History, Hypertension are categorical
    # We'll infer simple types
    continuous_cols = [
        "age",
        "weight",
        "height",
        "pulse",
        "systolic_bp",
        "diastolic_bp",
    ]
    # Filter to only those present
    continuous_cols = [c for c in continuous_cols if c in df_real.columns]

    print("\n--- 1. Univariate Fidelity (KS Test) ---")
    ks_results = calculate_ks_test(df_real, df_syn, continuous_cols)
    for col, res in ks_results.items():
        print(
            f"{col}: KS Stat = {res['KS_Stat']:.4f} (Lower is better), Score = {res['Score']:.4f}"
        )

    print("\n--- 2. Multivariate Fidelity (Correlation) ---")
    frob_norm, _, _ = calculate_correlation_diff(df_real, df_syn)
    print(
        f"Correlation Matrix Difference (Frobenius Norm): {frob_norm:.4f} (Lower is better)"
    )

    print("\n--- 3. Adversarial Fidelity (Discriminator) ---")
    auc = calculate_discriminator_score(df_real.copy(), df_syn.copy())
    print(f"Discriminator AUC: {auc:.4f}")
    print(
        "Interpretation: 0.5 = Perfect Fidelity (Indistinguishable), 1.0 = Poor Fidelity (Easy to spot)"
    )

    # Save Summary
    summary = {
        "Metric": ["Mean KS Score", "Correlation Diff", "Discriminator AUC"],
        "Value": [np.mean([r["Score"] for r in ks_results.values()]), frob_norm, auc],
    }
    pd.DataFrame(summary).to_csv("fidelity_results.csv", index=False)
    print("\nResults saved to fidelity_results.csv")


if __name__ == "__main__":
    data_path = "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/1_datasets/imputed_data.csv"
    run_fidelity_check(data_path)
