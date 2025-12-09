import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
import os
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation import train_generator

# Set style globally
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


def run_shap_analysis():
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "1_datasets", "encoded_data.csv")
    output_dir = os.path.join(base_dir, "2_data_analysis", "images")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # 2. Prepare Data
    # Drop target
    X = df.drop(columns=["hypertension"])
    y = df["hypertension"]

    # 3. Train Generator (Gaussian Copula) on Real Data
    print("Training Gaussian Copula on Real Data...")
    # Re-using the GeneratorWrapper from src.generation
    # Assuming the train_generator function returns a fitted synthesizer
    synthesizer = train_generator(df, model_type="copula")

    # 4. Generate Synthetic Data (50% of Real Size)
    n_real = len(df)
    n_syn = int(n_real * 0.5)
    print(f"Generating {n_syn} synthetic samples (50% augmentation)...")

    syn_data = synthesizer.sample(num_rows=n_syn)
    X_syn = syn_data.drop(columns=["hypertension"])
    y_syn = syn_data["hypertension"]

    # 5. Augment Data (Real + Synthetic)
    print("Augmenting data...")
    X_aug = pd.concat([X, X_syn], axis=0)
    y_aug = pd.concat([y, y_syn], axis=0)

    print(f"Training set size: {len(X_aug)} (Real: {len(X)} + Syn: {len(X_syn)})")

    # DEBUG: Check what the model is actually seeing
    print("DEBUG: First 5 rows of Augmented Training Data:")
    print(X_aug.head())
    print("DEBUG: Feature columns:", X_aug.columns.tolist())

    # 6. Train Random Forest (The "Winning" Model)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,  # Reasonable default
    )
    model.fit(X_aug, y_aug)

    # 7. Calculate SHAP Values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Explain the REAL data
    shap_values = explainer.shap_values(X)

    # Handle SHAP output format (Binary Classification)
    # shap_values can be:
    # 1. List of [Array(Samples, Features), Array(Samples, Features)] -> Select index [1]
    # 2. Array(Samples, Features, Classes) -> Select slice [:, :, 1]

    vals = np.array(shap_values)
    print(f"DEBUG: Feature columns: {X.columns.tolist()}")
    print(f"DEBUG: Raw SHAP output shape: {vals.shape}")

    if isinstance(shap_values, list):
        print("DEBUG: SHAP output is a list.")
        shap_values_target = shap_values[1]  # Class 1
    elif vals.ndim == 3:
        # (Samples, Features, Output) or (Samples, Output, Features)?
        # TreeExplainer usually outputs (Samples, Features, Classes) for binary
        print("DEBUG: SHAP output is 3D Array. Selecting Class 1.")
        # We assume last dim is classes. If Shape is (134, 8, 2) -> (Samples, Features, Classes)
        if vals.shape[-1] == 2:
            shap_values_target = vals[:, :, 1]
        else:
            # Fallback
            print("WARNING: Unsure of shape, using Class 0 reverse?")
            shap_values_target = vals[:, :, 1]
    else:
        # Already 2D?
        shap_values_target = shap_values

    # Double check shape
    print(f"DEBUG: Final Target Shape: {np.array(shap_values_target).shape}")

    # 8. Generate Plots

    # Plot 1: Summary Beeswarm Plot
    print("Generating SHAP Summary Plot (Beeswarm)...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_target,  # 2D Array
        X,
        show=False,
        cmap="cool",
        plot_size=(12, 10),
    )
    plt.title(
        "Feature Importance & Directionality (SHAP - Class: Hypertension)",
        fontsize=24,
        pad=20,
    )

    save_path_bees = os.path.join(output_dir, "shap_summary_beeswarm.png")
    plt.savefig(save_path_bees, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path_bees}")

    # Plot 2: Bar Plot (Absolute Importance)
    print("Generating SHAP Bar Plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_target,
        X,
        plot_type="bar",
        show=False,
        color="#00CED1",
        plot_size=(12, 10),
    )
    plt.title("Top Predictors of Hypertension (Mean |SHAP|)", fontsize=24, pad=20)

    save_path_bar = os.path.join(output_dir, "shap_summary_bar.png")
    plt.savefig(save_path_bar, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path_bar}")

    # ---------------------------------------------------------
    # PRINT TEXT SUMMARY FOR INTERPRETATION
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("SHAP FEATURE IMPORTANCE SUMMARY")
    print("=" * 50)

    # Calculate Mean Absolute SHAP (Magnitude)
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    feature_names = list(X.columns)

    print(f"Features: {len(feature_names)}, Scores: {len(mean_abs_shap)}")

    # Create DataFrame
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": mean_abs_shap}
    ).sort_values("Importance", ascending=False)

    print(f"{'Feature':<20} | {'Importance':<10} | {'Effect (Correlation)'}")
    print("-" * 55)

    for _, row in importance_df.iterrows():
        feature = row["Feature"]
        score = row["Importance"]

        try:
            idx = feature_names.index(feature)
            feat_vals = X[feature].values
            shap_vals = shap_values_target[:, idx]

            # Correlation
            corr = np.corrcoef(feat_vals.astype(float), shap_vals)[0, 1]

            direction = "Risk (+)" if corr > 0 else "Protective (-)"
            if abs(corr) < 0.1:
                direction = "Non-linear/Null"
        except Exception as e:
            direction = f"N/A ({str(e)})"

        print(f"{feature:<20} | {score:<10.4f} | {direction}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_shap_analysis()
