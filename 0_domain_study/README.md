# Domain Research

## Synthetic Data During COVID-19: A Case Study in Pandemic Response

### 1. The Challenge: Data Silos vs. Public Health Urgency

During the initial COVID-19 outbreak in 2020, researchers faced a critical
paradox: hospitals held massive amounts of patient data, but strict privacy
regulations (GDPR, HIPAA) prevented immediate sharing. Traditional
de-identification failed to mitigate re-identification risks, while cumbersome
legal reviews created "data silos" that delayed the collaborative research
needed to understand disease progression and resource allocation.

### 2. The Solution: The National COVID Cohort Collaborative (N3C)

To break these silos, the NIH’s National Center for Advancing Translational
Sciences (NCATS) launched the **National COVID Cohort Collaborative (N3C)**.

* **Scale:** N3C aggregated Electronic Health Records (EHR) from over 76 U.S.
institutions, encompassing 18.4 million individuals.
* **Methodology:** Using the OMOP common data model and MDClone technology, N3C
generated **synthetic derivatives**—computationally created "digital twins" that
replicated the statistical characteristics of real patients without containing a
single identifiable record.
* **The Breakthrough:** Because synthetic records are novel generations rather
than masked real data, they fell outside the scope of HIPAA/GDPR restrictions.
This allowed datasets to be shared globally without IRB approvals, enabling
rapid hypothesis testing.

### 3. Validation: The Foraker Study (2021)

To prove that synthetic data was not just safe but *useful*, Foraker et al.
(2021) conducted a rigorous validation across three distinct clinical use cases.
The results provided foundational evidence for the **Train on Synthetic, Test on
Real (TSTR)** paradigm.

* **Use Case 1: Cohort Characterization (Fidelity)**
Researchers compared 15 demographic and clinical characteristics (e.g., age,
race, comorbidities, labs) between synthetic and real cohorts. Univariate and
multivariate tests confirmed that synthetic distributions successfully mimicked
the original data ($p > 0.05$), proving that statistical properties were
preserved for exploratory analysis.

* **Use Case 2: Predictive Modeling (Utility/TSTR)**
The team trained machine learning models exclusively on synthetic data to
predict hospital admission risk for 230,703 COVID-positive patients. Using 11
clinical features—including age, gender, race, BMI, and O2 saturation—these
models were evaluated on held-out real data.
* **Result:** The models performed comparably to those trained on original
data, proving that algorithms developed on privacy-safe data generalize
effectively to real clinical populations.

* **Use Case 3: Geospatial Epidemiology (Temporal Integrity)**
Epidemic curves constructed from synthetic data (representing 1.85 million
tests) overlaid almost perfectly with real-world curves. The synthetic
generation maintained accurate temporal ordering without the date-shifting
distortions often required in traditional de-identification.

### 4. Impact: Democratization and Preparedness

The success of N3C and open-source platforms like **Synthea** (which simulated
ventilator and dialysis demand) democratized access to high-quality data.

* **Research Velocity:** Over 450 projects leveraged N3C data, resulting in
100+ peer-reviewed manuscripts on mortality prediction and treatment patterns.
* **Global Equity:** Small hospitals and resource-limited institutions could
develop algorithms and operational plans immediately, without waiting months for
legal data-sharing agreements.

> *While these large-scale initiatives proved the utility of synthetic data for
global pandemics, the same principles offer a unique lifeline for
resource-constrained environments. This project applies these validated
frameworks to address data poverty in a specific, high-risk demographic:
geriatric patients in Nigeria.*

## Regional Context: Synthetic Data as an Equity Tool in LMICs

### 1. The "Data Poverty" Gap in African Healthcare

While high-income countries leverage massive datasets to advance AI,
Sub-Saharan Africa faces a crisis of "data poverty." The region bears **25% of
the global disease burden** yet houses only **3% of the world's healthcare
professionals**. This disparity is exacerbated by severe infrastructural
challenges:

* **Digitization:** Over 60% of African health facilities rely on paper-based
records.
* **Infrastructure (Nigeria):** Electricity access stands at ~60.5% and
internet penetration at 40%, making the adoption of Electronic Health Records
(EHR) costly and inconsistent.
* **Fragmentation:** Data remains siloed across disease-specific systems (e.g.,
HIV, TB, malaria), meaning only **5% of available health data** is effectively
utilized for research.

### 2. The Consequence: Algorithmic Bias

The lack of local data forces African researchers to rely on models trained on
Western populations, leading to dangerous generalization failures.

* **Diagnostic Failure:** AI models trained on white populations frequently
fail to diagnose skin cancer on non-white skin.
* **Triage Bias:** Algorithms often underestimate risk in Black patients due to
proxies rooted in systemic inequities.
By developing models on external data, African institutions risk reproducing
Western-centric biases and obscuring local epidemiological realities.

### 3. The Regulatory Landscape: Nigeria Data Protection Act (2023)

The solution is not simply "more data sharing," as regulatory frameworks are
tightening. Nigeria recently enacted the **Nigeria Data Protection Act (NDPA)
2023**, establishing the Nigeria Data Protection Commission (NDPC) and enforcing
rigorous governance for sensitive health data. This aligns with a broader trend
where 36 of 54 African nations now have data protection laws. While necessary
for privacy, these laws create additional barriers to accessing the granular
data needed for AI development.

### 4. The Solution: Synthetic Data for Digital Sovereignty

Synthetic data generation offers a pragmatic path to "Digital Health
Sovereignty" in Nigeria.

* **Privacy Compliance:** By training generative models (e.g., CTGAN,
CopulaGAN) on limited real data, researchers can create synthetic cohorts that
preserve statistical properties without containing identifiable information,
bypassing strict NDPA/GDPR sharing restrictions.
* **Equity & Fairness:** Crucially, synthetic data allows for the deliberate
**oversampling** of underrepresented groups. Research shows that balancing
datasets synthetically can improve model fairness and performance by up to **31%
(ROCAUC)**.
* **Validation:** A recent proof-of-concept in Kenya’s health surveillance
system demonstrated that models like CTGAN could generate data with high
fidelity and utility, proving viability in the African context.

> *Synthetic data thus transcends privacy; it is a critical enabler of data
democratization. By mitigating the "cold start" problem for AI in Nigeria, this
project aims to empower locally relevant innovation for the populations that
need it most.*

## 3. Clinical Domain: Synthetic Augmentation for Hypertension

### 1. The Clinical Challenge: Imbalance and Small Sample Sizes

Developing robust predictive models for Cardiovascular Disease (CVD) and
hypertension is frequently hindered by two data limitations:

* **Class Imbalance:** Clinical datasets often exhibit severe ratios between
hypertensive and normotensive cases (e.g., 1:15). This forces Machine Learning
(ML) models to bias toward the majority class, resulting in poor sensitivity for
detecting at-risk patients.
* **Data Scarcity:** In specific demographics, such as geriatric populations,
datasets are often too small to support reliable risk estimation, leading to
overfitting where models memorize the training data rather than learning
generalizable physiological patterns.

### 2. Validated Strategies: From SMOTE to GANs

Research demonstrates that synthetic augmentation is not merely a "patch" but a
validated standard for enhancing cardiovascular prediction.

* **GAN-Based Success:** Recent studies (Sharma & Kumar, 2024) show that
Generative Adversarial Networks (GANs) can generate synthetic cardiovascular
patients that preserve statistical properties (cholesterol, BP). When training
on combined real and synthetic data, models achieved up to **99% accuracy**,
significantly outperforming baselines by providing diverse training examples.
* **SMOTE & Hybrid Methods:** Traditional interpolation techniques like
**SMOTE** (Synthetic Minority Oversampling Technique) have proven effective for
heart attack prediction. Newer hybrid approaches, such as **SMOTE-TOMEK** (Enihe
et al., 2025), which combines oversampling with noise removal, achieved a
**ROC-AUC of 96%** using LightGBM.
* **Equivalence:** Crucially, Arora & Arora (2023) demonstrated that models
trained on synthetic blood pressure data showed no significant difference in
error rates compared to real data, validating synthetic datasets as effective
training sources for clinical use.

### 3. The Geriatric Context: Modeling Non-Linearity

This project specifically targets an aged clinic, where the physiology of
hypertension differs markedly from the general population.

* **Non-Linearity:** Due to arterial stiffness and endothelial dysfunction, the
relationship between Age and Blood Pressure in the elderly is often
**non-linear** (e.g., U-shaped mortality curves).
* **The ML Advantage:** Standard regression models often fail to capture these
complexities. However, ensemble methods (e.g., AdaBoost) trained on augmented
data can successfully model these non-linear interactions, provided they have
enough data points to learn the curve.

> **Justification for Augmentation:**
> The evidence confirms that synthetic augmentation tackles class imbalance,
reduces overfitting in small geriatric cohorts, and improves key metrics
(F1-score, AUC). By incorporating population-specific heterogeneity, it allows
us to build equitable models for the "oldest old" that standard small datasets
cannot support.

## 4. Technical Challenge I: Imputation Techniques for High Missingness

### 1. The Pitfalls of Simple Imputation

Simple imputation methods, such as filling gaps with the **Mean** (for
continuous variables) or **Mode** (for categorical variables), are fundamentally
problematic for medical data analysis.

* **Systematic Bias:** Mean imputation assumes data is "Missing Completely at
Random" (MCAR), which is rarely true in clinical settings.
* **Variance Reduction:** By clustering values at the average, it distorts the
empirical distribution, artificially lowering variance. This produces overly
narrow confidence intervals, leading to false-positive findings (Type I errors).
* **Correlation Loss:** These methods treat variables in isolation, ignoring
the biological relationships between features (e.g., the link between BMI and
Hypertension).

### 2. Advanced Method A: MICE (The Gold Standard for Inference)

**Multiple Imputation by Chained Equations (MICE)** improves on simple methods
by iteratively modeling each incomplete variable conditionally on the others.

* **Mechanism:** It generates multiple plausible datasets to preserve
inter-variable relationships. This allows for valid uncertainty quantification
through Rubin’s pooling rules.
* **Flexibility:** MICE handles mixed data types using tailored conditional
models (e.g., linear regression for continuous, logistic regression for binary).
* **Limitation:** Parametric MICE assumes specific distributions (Gaussian) and
may miss complex non-linear interactions unless explicitly specified by an
expert.

### 3. Advanced Method B: Machine Learning Approaches

**KNN Imputation:**
Fills missing values based on the most similar observed records ("neighbors").
While non-parametric, it is sensitive to the choice of $k$, requires data
scaling, and struggles computationally with high-dimensional mixed data.

**MissForest:**
Uses Random Forest regression to iteratively predict missing values.

* **Strengths:** Effectively handles mixed data types and captures complex
**non-linear interactions** without explicit model specification. It is
excellent for predictive tasks.
* **Weaknesses:** It produces single imputations (point estimates), which lacks
the proper uncertainty representation found in MICE, potentially leading to
underestimated confidence intervals.

### 4. Comparative Evaluation

The literature presents a trade-off between *prediction* and *inference*:

* **Stekhoven & Bühlmann (2012)** introduced MissForest, demonstrating its
ability to handle complex interactions in mixed datasets.
* **Shah et al. (2014)**, however, demonstrated that **Random Forest-enhanced
MICE** yields more efficient and unbiased parameter estimates than parametric
MICE or MissForest alone, particularly when non-linearities exist.

### Summary of Techniques

<!-- markdownlint-disable MD013 -->
| Characteristic | Mean/Mode | MICE | KNN | MissForest |
| :--- | :--- | :--- | :--- | :--- |
| **Mixed Data Support** | Limited | **Excellent** | Moderate (requires dummy coding) | **Excellent** |
| **Variance Preservation** | Poor | **Good** | Moderate | Good |
| **Captures Non-Linearity** | No | Limited (Parametric) | Yes | **Yes** |
| **Multiple Imputation** | No | **Yes** | No | No |
| **Confidence Intervals** | Biased (Too Narrow) | **Valid** | Valid (if $k$ well chosen) | Biased (Too Narrow) |
| **Suitability** | Unsuitable | **Gold Standard** | Specific Cases | Exploratory/Predictive |
<!-- markdownlint-enable MD013 -->

> **Recommendation:**
> For high missingness in mixed medical tabular data, **MICE is the gold
standard** for preserving statistical validity. While MissForest is powerful for
pure prediction, MICE (or Random Forest MICE) ensures that the uncertainty of
the missing data is properly modeled, which is critical for valid clinical
inference.

## 5. Technical Challenge II: Generative Strategies (Copulas vs. CTGAN)

### 1. The Generative Landscape

Generating synthetic tabular health data requires a delicate balance between
**statistical fidelity**, **privacy preservation**, and **computational
efficiency**. For this project, we compare two leading methodologies
representing opposite ends of the complexity spectrum:

* **Gaussian Copulas:** A classical statistical approach relying on correlation
structures.
* **CTGAN (Conditional Tabular GAN):** A modern deep learning method based on
Generative Adversarial Networks.

### 2. The Statistical Approach: Gaussian Copulas

Gaussian Copulas function by decomposing the multivariate joint distribution of
the data into two components: the **marginal distributions** of each variable
and their **dependence structure** (captured via a correlation matrix).

* **Mechanism:** Based on Sklar's Theorem, it uses Spearman rank correlation to
model dependencies, ensuring invariance under distributional transformations.
Synthesis involves sampling from a multivariate normal distribution and
transforming samples back through learned marginal quantiles.
* **Advantages:** This method is **non-iterative**, meaning it does not "learn"
via epochs. This makes it extremely stable and reproducible even with very small
samples ($N < 200$). It offers robust privacy by summarizing population-level
correlations rather than memorizing individual records.

### 3. The Deep Learning Approach: CTGAN

CTGAN addresses the complexities of modern medical data (mixed variable types
and multi-modal distributions) using deep learning.

* **Mode-Specific Normalization:** CTGAN decomposes continuous variables using
a **Variational Gaussian Mixture Model (VGM)**. This allows it to capture
complex, multi-modal distributions (e.g., a bimodal distribution of blood
pressure) that simple statistical models might miss.
* **Conditional Generator:** To combat **mode collapse** (where the generator
ignores minority classes), CTGAN employs a conditional generator that samples
categorical levels based on log-frequency, forcing the model to learn rare
categories.
* **Architecture:** It utilizes Wasserstein GANs with gradient penalty
(WGAN-GP) to improve training stability.

### 4. The "Small Data" Wall

While CTGAN excels at capturing complex non-linear dependencies in large
datasets ($N > 10,000$), it faces severe limitations in low-data regimes.

* **Discriminator Overfitting:** Research indicates that CTGAN performance
deteriorates sharply below $\approx 640$ samples. In small datasets ($N < 200$),
the discriminator easily memorizes the real data, forcing the generator to
either replicate training data (privacy breach) or produce noise (utility loss).
* **Stability:** Comparative studies (Demetrio et al., 2024) show that Gaussian
Copulas maintain consistent utility and fidelity across sample sizes from 50 to
1000, whereas CTGAN correlations degrade dramatically in this range.

### 5. Selection Framework and Recommendation

The literature suggests a clear trade-off based on sample size and complexity:

<!-- markdownlint-disable MD013 -->
| Feature | Gaussian Copula | CTGAN |
| :--- | :--- | :--- |
| **Best for Sample Size** | $N < 1,000$ (Ideal for $N < 200$) | $N > 5,000$ |
| **Computational Cost** | Low (Fast, CPU-based) | High (Slow, GPU-based) |
| **Privacy Risk** | Low (Summarizes correlations) | Moderate (Risk of memorization) |
| **Interpretability** | High (Transparent correlations) | Low (Black box) |
<!-- markdownlint-enable MD013 -->

> **Recommendation:**
> Given the specific constraints of this project ($N=134$), **Gaussian
Copulas** are the preferred method. They offer simplicity, regulatory
transparency, and stability without the hyperparameter tuning required for GANs.
CTGAN serves as a comparative baseline to demonstrate the limitations of deep
learning in data-scarce medical environments.

## 6. Evaluation Framework: The "Trinity" of Validation

### 1. The Integrated Framework

Evaluating synthetic health data requires a multidimensional assessment
strategy that goes beyond simple accuracy. We employ a rigorous "Trinity"
framework spanning **Fidelity** (statistical likeness), **Utility** (downstream
performance), and **Privacy** (data safety). Balancing these three pillars is
the fundamental challenge of synthetic data generation, as enhancements in one
dimension often degrade the others.

### 2. Fidelity: Statistical Mirroring

Fidelity measures how well the synthetic data mirrors the real data, preserving
marginal distributions, joint dependencies, and higher-order structures
essential for clinical validity.

* **Univariate Tests:** We use the **Kolmogorov–Smirnov (KS) test** to check
distribution alignment for continuous variables (e.g., Age, BMI) and
**Chi-Squared tests** for categorical variables.
* **Multivariate Dependencies:** Feature interdependencies are assessed using
**Pearson** and **Spearman** correlation coefficients.
* **Divergence Metrics:** We quantify distributional discrepancies using
**Kullback–Leibler (KL)** and **Jensen–Shannon** divergence.
* **Classifier-Based Discrimination:** A "distinguisher" model (e.g., Random
Forest) is trained to classify records as Real vs. Synthetic. A *lower*
classification success rate indicates higher fidelity (the generator
successfully "fooled" the discriminator).

### 3. Utility: The TSTR Paradigm

Utility evaluates the synthetic data’s effectiveness in supporting valid
Machine Learning inference. The gold standard methodology is **Train on
Synthetic, Test on Real (TSTR)** (Esteban et al., 2017).

* **Methodology:** Models are trained exclusively on synthetic data and tested
on held-out real data.
* **Metrics:** Performance is measured via Accuracy, AUROC, F1-Score, and RMSE.
* **Thresholds:** A performance drop of **< 5–10%** compared to models trained
on real data (TRTR) is typically considered acceptable for research utility.
This metric is critical because it accounts for subtle distributional mismatches
that univariate fidelity tests might miss.

### 4. Privacy: Re-identification and Leakage

Privacy evaluation addresses the risks of memorization and attribute
disclosure.

* **Distance to Closest Record (DCR):** The primary metric for this project. It
measures the Euclidean distance between synthetic records and real records. A
DCR of 0 indicates the model has "memorized" a patient, creating a privacy
breach.
* **Membership Inference Attacks (MIA):** The empirical gold standard.
Adversaries attempt to infer if a specific patient was in the training set.
While fully synthetic data generally resists MIAs better than anonymized data,
advanced attacks (e.g., DOMIAS) can detect local overfitting.
* **Emerging Research:** While DCR is a standard "first line of defense,"
recent studies (Yao et al., 2025) warn that DCR alone may not capture all
leakage risks, suggesting the need for complementary formal frameworks like
**Differential Privacy**.

> **Project Application:**
> For this study, we prioritize **TSTR (Utility)** to prove that synthetic
augmentation improves hypertension prediction in geriatric cohorts, while
monitoring **DCR (Privacy)** to ensure compliance with the "Digital Twin" safety
standards required by the Nigeria Data Protection Act.

### References

1. **Foraker et al. (2021).** The National COVID Cohort Collaborative: Analyses
of a COVID-19 Registry. *JMIR Public Health and Surveillance*, 7(10), e30697.
2. **Washington University School of Medicine. (2022).** Synthetic data mimics
real patient data, accurately models COVID-19 pandemic.
3. **Walonoski et al. (2020).** Synthea Novel coronavirus (COVID-19) model.
*MedRxiv preprint*.
4. **Mwigereri et al. (2025).** Synthetic data generation of health and
demographic surveillance systems data in Kenya. *JMIR*.
5. **Okonkwo et al. (2025).** AI and African Health Equity. *APAS Proceedings*.
6. **SecurePrivacy AI. (2025).** Nigeria Data Protection Law: NDPA 2023
Overview.
7. **WEF (2025).** Digital Health Tools to Reduce Inequity in LMICs.
8. **Sharma & Kumar (2024).** Early Heart Disease Prediction Using
GAN-Augmented Data. *Computer Science and Information Technology*.
9. **Enihe et al. (2025).** The Effect of Imbalance Data Mitigation Techniques
on Cardiovascular Disease Prediction. *Nigerian Journal of Physiological
Sciences*.
10. **Hwang et al. (2024).** Machine Learning–Based Hypertension Prediction
with SMOTE Augmentation. *JMIR Medical Informatics*.
11. **Arora & Arora (2023).** Machine Learning Models Trained on Synthetic
Datasets for Blood Pressure Prediction. *PLOS ONE*.
12. **Sun et al. (2021).** The Non-Linear Relationship Between Systolic Blood
Pressure and Mortality in the Elderly. *Frontiers in Cardiovascular Medicine*.
13. **Van Buuren & Groothuis-Oudshoorn (2011).** mice: Multivariate Imputation
by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67.
14. **Sterne et al. (2009).** Multiple imputation for missing data in
epidemiological and clinical research: Potential and pitfalls. *BMJ*, 338,
b2393.
15. **Stekhoven & Bühlmann (2012).** MissForest—non-parametric missing value
imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118.
16. **Shah et al. (2014).** Comparison of Random Forest and Parametric
Imputation Models for Imputing Missing Data Using MICE. *American Journal of
Epidemiology*, 179(6), 764–774.
17. **Xu et al. (2019).** Modeling Tabular Data using Conditional GAN.
*Advances in Neural Information Processing Systems (NeurIPS)*, 32, 7335–7345.
18. **Patki et al. (2016).** The Synthetic Data Vault. *IEEE International
Conference on Data Science and Advanced Analytics (DSAA)*, 399–410.
19. **Demetrio et al. (2024).** A 'Marginally' Better CTGAN for the Low Sample
Regime. *DAGM-GCPR 2023 Proceedings*.
20. **Sklar, A. (1959).** Fonctions de répartition à n dimensions et leurs
marges. *Publications de l'Institut de Statistique de l'Université de Paris*, 8,
229–231.
21. **Callahan & MacNeill (2025).** Synthetic Tabular Data Generation for
Privacy-Preserving Machine Learning. *Journal of Computer Science and Statistics
Applications*, 5(7).
22. **Hernandez et al. (2025).** Comprehensive Evaluation Framework for
Synthetic Tabular Data. *Nature Medicine*, 31(4), 892–901.
23. **Esteban, C., Hyland, S., & Rätsch, G. (2017).** Real-Valued (Medical) Time
Series Generation with Recurrent Conditional GANs. *arXiv preprint
arXiv:1706.02633*.
24. **Goncalves et al. (2020).** Generation and Evaluation of Synthetic Patient
Data. *Scientific Data*, 7(1), 130.
25. **Yao et al. (2025).** The DCR Delusion: Measuring the Privacy Risk of
Synthetic Data. *arXiv preprint arXiv:2505.01524*.
26. **Lehman et al. (2021).** Membership Inference Attacks Against Synthetic
Health Data. *JAMIA*, 28(12), 2654–2662.
27. **Zamzmi et al. (2025).** A Scorecard for Synthetic Medical Data Evaluation.
*Nature Communications*, 16, 7820.
