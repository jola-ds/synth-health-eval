# Source Code Documentation

This directory contains the core Python modules that power the synthetic data
evaluation pipeline.

## Core Modules

### 1. Data Preparation

* **`imputation.py`**: Implements the "Imputation Shootout". Contains classes
for MICE, KNN, and MissForest imputation, along with evaluation logic to select
the best method.
* **`check_outliers.py`**: Utility script to detect and summarize outliers in
the dataset using the IQR method.
* **`visualize_data.py`**: Generates exploratory visualizations (scatter plots,
heatmaps) for the dataset documentation.

### 2. Generation

* **`generation.py`**: Defines the `GeneratorWrapper` class. This provides a
unified interface for different synthetic data generators (Gaussian Copula,
CTGAN), handling training, sampling, and persistence.

### 3. Evaluation & Analysis

* **`evaluation.py`**: The "Master Loop". Implements the Repeated Stratified
K-Fold Cross-Validation logic. It trains models on Real, Synthetic, and
Augmented data (Scenarios A, B, C, D) and records performance metrics.
* **`fidelity.py`**: Performs statistical fidelity checks. Calculates KS Test
scores (univariate shape), Correlation Matrix differences (multivariate
structure), and Discriminator AUC (adversarial realism).
* **`analysis.py`**: Aggregates raw results from the Master Loop into summary
tables (Mean/Std). Handles both Gaussian Copula and CTGAN results automatically.

### 4. Production

* **`train_final_model.py`**: Trains the final production model using the
winning strategy (Random Forest + Gaussian Copula Augmentation) and saves it as
a pickle file.

## Usage

These modules are designed to be imported by the notebooks in `3_notebooks/` or
run directly for specific tasks.

Example:

```python
from src.generation import GeneratorWrapper
gen = GeneratorWrapper(model_type='copula')
gen.fit(data)
synthetic_data = gen.sample(100)
```
