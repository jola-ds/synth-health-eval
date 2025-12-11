# Experimental Results

This directory contains all the raw and aggregated data generated during the
evaluation of the synthetic data generators.

## File Descriptions

### 1. Master Loop Results (Raw Data)

These files contain the detailed, fold-by-fold performance metrics for every
run of the evaluation loop (5 Folds x 5 Repeats = 25 Runs per Scenario).

* **`results_master_loop_copula.csv`**: Results for the **Gaussian Copula**
generator. Includes F1, Accuracy, and AUC for Scenarios A (Baseline), B
(Fidelity), C (Scale), and D (Augmentation).
* **`results_master_loop_ctgan.csv`**: Results for the **CTGAN** generator.
Shows the performance drop due to convergence issues on small data.
* **`results_master_loop_params.csv`**: Results from the hyperparameter tuning
phase (using Copula).

### 2. Aggregated Summaries

These files provide the mean and standard deviation of the metrics across all
folds, useful for quick analysis and plotting.

* **`results_summary_copula.csv`**: Summary statistics for Gaussian Copula.
* **`results_summary_ctgan.csv`**: Summary statistics for CTGAN.

### 3. Fidelity & Tuning

* **`fidelity_results.csv`**: Statistical fidelity metrics comparing Real vs.
Synthetic data.
* *KS Score:* Univariate distribution similarity.
* *Correlation Diff:* Multivariate structure preservation.
* *Discriminator AUC:* Adversarial indistinguishability.
* **`results_params.csv`**: The optimal hyperparameters selected for the final
Random Forest model.

## Usage

These CSVs are referenced by the [analysis_report.md](../analysis_report.md) and
[methodology_report.md](../methodology_report.md) to substantiate the findings.
You can load them into pandas for further custom analysis:

```python
import pandas as pd
df = pd.read_csv("2_data_analysis/results/results_summary_copula.csv")
print(df.head())
```
