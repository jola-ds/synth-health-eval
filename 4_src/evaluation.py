import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.generation import train_generator
from xgboost import XGBClassifier

# Define Predictors and their Hyperparameter Grids
PREDICTORS = {
    "RF": (
        RandomForestClassifier(random_state=42),
        {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
    ),
    "XGB": (
        XGBClassifier(random_state=42, eval_metric="logloss"),
        {
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__max_depth": [3, 5, 7],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]},
    ),
    "LR": (
        LogisticRegression(random_state=42, max_iter=1000),
        {"model__C": [0.1, 1.0, 10.0], "model__solver": ["liblinear", "lbfgs"]},
    ),
}


def train_and_evaluate(
    X_train, y_train, X_test, y_test, model_name, base_model, param_grid
):
    """
    Trains a model using RandomizedSearchCV and evaluates on test set.
    """
    # Scale data for KNN/LR/SVM
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

    # Tuning (Internal 3-Fold CV)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=5,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)
    y_prob = (
        best_model.predict_proba(X_test)[:, 1]
        if hasattr(best_model, "predict_proba")
        else [0] * len(y_test)
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5,
        "Best_Params": search.best_params_,
    }


def run_master_loop(
    data_path, output_path, params_output_path=None, model_type="copula"
):
    """
    Executes the 25-run Repeated Stratified K-Fold loop.
    """
    print(f"Starting Master Loop with {model_type.upper()}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["hypertension"])
    y = df["hypertension"]

    # Repeated Stratified K-Fold (5 Splits x 5 Repeats = 25 Runs)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    all_results = []

    fold_idx = 0
    for train_index, test_index in rskf.split(X, y):
        fold_idx += 1
        print(f"Processing Fold {fold_idx}/25...")

        # 1. Split Data
        X_train_real, X_test_real = X.iloc[train_index], X.iloc[test_index]
        y_train_real, y_test_real = y.iloc[train_index], y.iloc[test_index]

        # 2. Train Generator on Real Train Fold
        train_fold_df = pd.concat([X_train_real, y_train_real], axis=1)

        # Use the unified train_generator function
        synthesizer = train_generator(train_fold_df, model_type=model_type)

        # 3. Generate Synthetic Pools
        # Pool for Scenario B (Same size as Real Train)
        syn_pool_B = synthesizer.sample(num_rows=len(train_fold_df))
        X_syn_B = syn_pool_B.drop(columns=["hypertension"])
        y_syn_B = syn_pool_B["hypertension"]

        # Pool for Scenario C (Max Pool = 1000)
        syn_pool_C = synthesizer.sample(num_rows=1000)
        X_syn_C = syn_pool_C.drop(columns=["hypertension"])
        y_syn_C = syn_pool_C["hypertension"]

        # 4. Run Scenarios for each Predictor
        for name, (model, grid) in PREDICTORS.items():
            # Scenario A: Baseline (Real Train)
            res_A = train_and_evaluate(
                X_train_real, y_train_real, X_test_real, y_test_real, name, model, grid
            )
            res_A.update({"Fold": fold_idx, "Scenario": "A_Baseline"})
            all_results.append(res_A)

            # Scenario B: Fidelity (Syn Train = Real Size)
            res_B = train_and_evaluate(
                X_syn_B, y_syn_B, X_test_real, y_test_real, name, model, grid
            )
            res_B.update({"Fold": fold_idx, "Scenario": "B_Fidelity"})
            all_results.append(res_B)

            # Scenario C: Scale (Syn Train = 1000)
            res_C = train_and_evaluate(
                X_syn_C, y_syn_C, X_test_real, y_test_real, name, model, grid
            )
            res_C.update({"Fold": fold_idx, "Scenario": "C_Scale"})
            all_results.append(res_C)

            # Scenario D: Augmentation (Real + Syn Ratios)
            ratios = [0.5, 1.0, 2.0, 4.0]
            for ratio in ratios:
                n_aug = int(len(train_fold_df) * ratio)
                syn_aug = synthesizer.sample(num_rows=n_aug)

                X_aug = pd.concat(
                    [X_train_real, syn_aug.drop(columns=["hypertension"])]
                )
                y_aug = pd.concat([y_train_real, syn_aug["hypertension"]])

                res_D = train_and_evaluate(
                    X_aug, y_aug, X_test_real, y_test_real, name, model, grid
                )
                res_D.update({"Fold": fold_idx, "Scenario": f"D_Augment_{ratio}"})
                all_results.append(res_D)

    # Save Results
    results_df = pd.DataFrame(all_results)

    # Separate Params if requested
    if params_output_path:
        # Extract params into a separate dataframe
        params_data = []
        for res in all_results:
            row = {
                "Fold": res["Fold"],
                "Scenario": res["Scenario"],
                "Model": res["Model"],
                **res["Best_Params"],  # Flatten the dict
            }
            params_data.append(row)

        params_df = pd.DataFrame(params_data)
        params_df.to_csv(params_output_path, index=False)
        print(f"Hyperparameters saved to {params_output_path}")

        # Remove params column from main results to keep it clean
        results_df = results_df.drop(columns=["Best_Params"])

    results_df.to_csv(output_path, index=False)
    print(f"Master Loop Complete! Results saved to {output_path}")


if __name__ == "__main__":
    import os

    # Define paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    data_path = os.path.join(project_root, "1_datasets", "imputed_data.csv")
    output_path = os.path.join(project_root, "results_master_loop_params.csv")
    params_path = os.path.join(project_root, "results_params.csv")

    # Run with Copula (since it was the winner) to get the final params
    run_master_loop(
        data_path, output_path, params_output_path=params_path, model_type="copula"
    )
