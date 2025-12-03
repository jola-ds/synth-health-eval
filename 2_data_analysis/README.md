# Data Analysis & Evaluation

This directory contains the final analysis, methodology, and raw results of the
synthetic data evaluation.

## Contents

### 1. Reports

* **[Methodology](methodology.md):** Detailed explanation of the data pipeline,
imputation strategy, generator selection, and evaluation metrics.
* **[Analysis Report](analysis_report.md):** The final findings, including
performance tables, head-to-head comparisons, and recommendations.

### 2. Results

* **[Results Directory](results/):** Contains all raw CSV output files from the
evaluation loop and fidelity checks.

## Key Findings Summary

* **Best Generator:** Gaussian Copula (Statistical)
* **Best Strategy:** Augmentation (50% Synthetic + 100% Real)
* **Outcome:** Maintained baseline F1-Score (0.825) while adding data
robustness. CTGAN failed to converge on this small dataset.
