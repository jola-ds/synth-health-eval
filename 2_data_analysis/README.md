# Data Analysis & Evaluation

This directory contains the final analysis, methodology, and raw results of the
synthetic data evaluation.

## Contents

### 1. Reports

* **[Methodology](methodology.md):** Detailed explanation of the data pipeline,
imputation strategy, generator selection, and evaluation metrics.
* **[Analysis Report](analysis_report.md):** The final findings + interpretation,
including performance tables, head-to-head comparisons, and recommendations.

### 2. Results

* **[Results Directory](results/):** Contains all raw CSV output files from the
evaluation loop and fidelity checks.

## Result Summary

*Performance of Random Forest with Gaussian Copula Augmentation.*

<!-- markdownlint-disable MD013 -->
| Scenario | Description | F1-Score | Accuracy | AUC | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Baseline** | Real Data Only | 0.826 | 0.842 | 0.898 | Gold Standard |
| **B. Fidelity** | Synthetic Only | 0.727 | 0.744 | 0.812 | Good Approximation |
| **D. Augment** | Real + 50% Syn | **0.825** | **0.841** | **0.891** |**Matches Baseline**|
<!-- markdownlint-enable MD013 -->
