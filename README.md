# Early Dropout Risk Prediction in Psychotherapy

## Abstract
This project builds an interpretable machine learning pipeline to estimate early psychotherapy dropout risk from clinically meaningful session-level features. It includes data generation/loading, feature engineering, model training (Logistic Regression, Random Forest, XGBoost), SHAP-based interpretability, and deployment interfaces (Streamlit + FastAPI). The current research setup uses clinically informed synthetic data for training and evaluation, with DAIC-WOZ used only to validate feature selection logic.

## Why Synthetic Data
- There is no broadly accepted public dataset that directly captures psychotherapy dropout outcomes with rich, reusable clinical session structure for this exact task.
- This is a known practical research gap: available public mental-health datasets often focus on depression detection rather than treatment dropout trajectories.
- To avoid target mismatch and leakage, this project uses synthetic data generated from published evidence:
  - Approximate dropout prevalence range in psychotherapy settings (~20-50% reported across studies/settings), with a 35% synthetic base rate used in experiments.
  - Predictor families consistently cited in dropout literature: symptom severity, attendance/engagement consistency, and session gaps.
- DAIC-WOZ (189 participants in the local split files) is retained for feature-selection sanity checks and exploratory validation, not as the training source for the dropout model.

## Dataset
- **Primary training/evaluation dataset:** synthetic cohort (default 500 patients, ~35% dropout rate).
- **Synthetic features:** `phq9_score`, `session_number`, `session_frequency_per_month`, `attendance_consistency`, `gap_between_sessions_days`, `mood_rating`, `age`, `phq9_change_rate`, plus engineered engagement features.
- **Reference dataset:** DAIC-WOZ (local raw CSVs) used for feature validation only.

## Model Pipeline
1. Generate/load data (`src/data_loader.py`)
2. Engineer dropout-relevant features (`src/feature_engineering.py`)
3. Split/scale + class balancing with SMOTE (`src/model.py`)
4. Train and compare:
   - Logistic Regression (baseline)
   - Random Forest (non-linear ensemble)
   - XGBoost (boosted trees)
5. Evaluate with:
   - AUC-ROC
   - F1
   - Precision
   - Recall

## Interpretability
- SHAP is used for both global and local interpretability:
  - Global summary plots (population-level risk drivers)
  - Individual waterfall plots (patient-level contribution breakdown)
- This supports transparent, clinician-readable reasoning rather than opaque scores.

## Results
To be filled after final retraining and locked experimental runs.

## Tech Stack
- Python
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- FastAPI
- Pandas / NumPy / Matplotlib / Seaborn

## Ethical Disclaimer
This repository is a research prototype for educational and methodological exploration. It is **not** a medical device, is **not** validated for clinical deployment, and must not be used as a standalone basis for diagnosis, treatment, or triage decisions.

## Citation
- Swift, J. K., & Greenberg, R. P. (2012). Premature discontinuation in adult psychotherapy.
- McGovern, et al. (2024). Contemporary evidence on psychotherapy dropout predictors.
- Lutz, et al. (2022). Session-level monitoring and psychotherapy outcome/dropout risk factors.
- Gratch, et al. (2014). The Distress Analysis Interview Corpus (DAIC-WOZ).
