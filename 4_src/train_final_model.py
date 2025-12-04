import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.generation import train_generator


def get_best_params(params_path, model_name="RF", scenario="D_Augment_0.5"):
    """
    Finds the most frequent hyperparameter combination for a given model and scenario.
    """
    df = pd.read_csv(params_path)

    # Filter
    subset = df[(df["Model"] == model_name) & (df["Scenario"] == scenario)]

    if subset.empty:
        raise ValueError(f"No params found for {model_name} in {scenario}")

    # We want to find the mode of the parameters.
    # Since params are in columns, we can group by all param columns and count.
    all_param_cols = [c for c in subset.columns if c.startswith("model__")]

    # 1. Keep only columns relevant to this model (not all NaNs)
    # We check if at least one value is not NaN.
    # Note: Some valid params might be NaN (e.g. max_depth=None), so we need to be careful.
    # But params for OTHER models (e.g. learning_rate for RF) will be ALL NaN.
    param_cols = [c for c in all_param_cols if subset[c].notna().any()]

    # 2. Fill NaNs with a placeholder so groupby doesn't drop them
    # (Pandas groupby drops NaNs by default)
    subset_filled = subset.copy()
    subset_filled[param_cols] = subset_filled[param_cols].fillna("None_Value")

    # Count frequency of each combination
    mode_row = (
        subset_filled.groupby(param_cols)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .iloc[0]
    )

    print(
        f"Most frequent parameters for {model_name} ({scenario}) found in {mode_row['count']} folds:"
    )
    best_params = {}
    for col in param_cols:
        val = mode_row[col]

        # Convert back 'None_Value' to None
        if val == "None_Value":
            val = None
        elif isinstance(val, (float, np.floating)) and float(val).is_integer():
            val = int(val)

        print(f"  {col}: {val} ({type(val)})")
        # Strip 'model__' prefix for actual initialization
        param_name = col.replace("model__", "")
        best_params[param_name] = val

    return best_params


def train_final_model(data_path, params_path, output_model_path):
    print("Training Final Model...")

    # 1. Load Real Data
    df = pd.read_csv(data_path)
    X_real = df.drop(columns=["hypertension"])
    y_real = df["hypertension"]

    # 2. Get Best Params
    # We chose RF and Augmentation 0.5 as the winner
    rf_params = get_best_params(params_path, model_name="RF", scenario="D_Augment_0.5")

    # 3. Generate Synthetic Data for Augmentation
    print("Generating synthetic data for augmentation...")
    # Train generator on ALL real data
    gen = train_generator(df, model_type="copula")

    # Generate 50% of real size
    n_aug = int(len(df) * 0.5)
    syn_data = gen.sample(n_aug)
    X_syn = syn_data.drop(columns=["hypertension"])
    y_syn = syn_data["hypertension"]

    # Combine
    X_final = pd.concat([X_real, X_syn])
    y_final = pd.concat([y_real, y_syn])

    print(
        f"Final Training Set: {len(X_final)} samples (Real: {len(X_real)}, Syn: {len(X_syn)})"
    )

    # 4. Train Model
    # Note: We use a Pipeline with StandardScaler because that's what we evaluated,
    # even though RF doesn't strictly need scaling, consistency is key.
    model = RandomForestClassifier(random_state=42, **rf_params)

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

    pipeline.fit(X_final, y_final)

    # 5. Save
    joblib.dump(pipeline, output_model_path)
    print(f"Final Model saved to {output_model_path}")


if __name__ == "__main__":
    data_path = "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/1_datasets/imputed_data.csv"
    params_path = (
        "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/results_params.csv"
    )
    output_model_path = (
        "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/final_model_rf.pkl"
    )

    train_final_model(data_path, params_path, output_model_path)
