import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Loads data."""
    df = pd.read_csv(filepath)
    return df


def get_imputers():
    """Returns a dictionary of imputers to test."""
    return {
        "MICE": IterativeImputer(random_state=42),
        "KNN": KNNImputer(n_neighbors=5),
        "MissForest": IterativeImputer(
            estimator=RandomForestRegressor(n_jobs=-1), max_iter=10, random_state=42
        ),
    }


def evaluate_imputers(df, target_col="hypertension"):
    """
    Runs 5-Fold CV on the dataset using different imputers.
    Returns a results dataframe.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identify columns with missing values
    missing_cols = X.columns[X.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_cols}")

    imputers = get_imputers()
    results = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, imputer in imputers.items():
        print(f"Testing Imputer: {name}...")

        # Pipeline: Impute -> Scale -> Classify
        # Use a simple RF Classifier to judge the quality of imputation for the downstream task
        pipeline = Pipeline(
            [
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )

        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

        # Check distribution preservation
        # This is a heuristic check
        imputer.fit(X)
        X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns)

        dist_diff = 0
        for col in missing_cols:
            orig_mean = X[col].mean()
            imp_mean = X_imputed[col].mean()
            dist_diff += abs(orig_mean - imp_mean)

        results.append(
            {
                "Imputer": name,
                "CV_Accuracy": np.mean(scores),
                "CV_Std": np.std(scores),
                "Mean_Shift": dist_diff,  # Lower is better (usually, unless missingness is MNAR)
            }
        )

    return pd.DataFrame(results)


def save_imputed_data(df, output_path):
    """
    Imputes the dataset using winner, MICE, and saves it to CSV.
    """
    print("Imputing full dataset with MICE...")
    imputer = IterativeImputer(random_state=42)
    # Fit and transform
    # IterativeImputer returns a numpy array, we need to wrap it back in DataFrame
    imputed_values = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed_values, columns=df.columns)

    # Round specific columns
    binary_cols = ["gender", "hypertension", "history_hypertension"]
    for col in binary_cols:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].round()

    df_imputed.to_csv(output_path, index=False)
    print(f"Imputed data saved to: {output_path}")


if __name__ == "__main__":
    # Load data
    input_path = "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/1_datasets/encoded_data.csv"
    output_path = "c:/Users/Moses Omotunde/Documents/Me/synth-health-eval/1_datasets/imputed_data.csv"

    df = load_data(input_path)

    # Run Shootout
    results = evaluate_imputers(df)
    print("\nImputation Shootout Results:")
    print(results)

    # Save MICE Imputed Data (The Winner)
    save_imputed_data(df, output_path)
