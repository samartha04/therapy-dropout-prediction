# 🧠 Early Dropout Risk Prediction in Psychotherapy

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-blue.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-blue.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.svg)](https://streamlit.io/)

A machine learning pipeline designed to estimate early psychotherapy dropout risk using clinically meaningful session-level features. This project heavily focuses on **interpretability** so that clinicians can understand *why* a particular risk score is assigned to a patient, rather than relying on an opaque model.

---

## 📖 Abstract

This repository provides an end-to-end framework, including:
1. **Data Generation & Loading**: Tools to construct synthetic clinically-aligned datasets.
2. **Feature Engineering**: Derivation of session-engagement metrics and symptom change rates.
3. **Model Training & Evaluation**: Comparison across Logistic Regression, Random Forest, and XGBoost models.
4. **SHAP-based Interpretability**: Providing global feature importance and patient-level risk contributors.
5. **Deployment Interfaces**: A FastAPI backend and a Streamlit frontend for interactive local exploration.

*(Note: The current research setup uses clinically informed synthetic data for training and evaluation. The DAIC-WOZ dataset is referenced solely to validate feature selection logic.)*

---

## 🧐 Why Synthetic Data?

* **Missing Real-World Datasets**: There is no broadly accepted public dataset that directly captures psychotherapy dropout outcomes with rich, reusable clinical session structure. Available public mental-health datasets often focus on depression detection rather than treatment dropout trajectories.
* **Avoiding Target Leakage**: To avoid target mismatch, this project relies on synthetic data generated from published evidence.
  * Captures an approximate dropout prevalence range (~20-50% reported across studies/settings). Our standard experiments use a **35% synthetic base dropout rate**.
  * Employs predictor families consistently cited in dropout literature: *symptom severity, attendance/engagement consistency, and inter-session gaps*.

---

## 🗂️ Dataset & Features

* **Primary Training/Evaluation**: Synthetic cohort (default 500 patients, ~35% dropout rate).
* **Clinical Features Traced**:
  * `phq9_score`: Patient's current depression severity.
  * `session_number`: Sequence in the treatment protocol.
  * `session_frequency_per_month`: Number of appointments managed per month.
  * `attendance_consistency`: Metric denoting the reliability of attending scheduled sessions.
  * `gap_between_sessions_days`: Disruption timeline between consultations.
  * `mood_rating` & `age`.
  * `phq9_change_rate`: Engineered rate representing symptomatic trajectory.
* **Reference Data**: DAIC-WOZ (local raw CSVs) used exclusively for cross-referencing and feature validation.

---

## 🚀 Model Pipeline

The ML workflow is modularly organized in the `src/` directory:

1. **`src/data_loader.py`**: Generate/load target dataset.
2. **`src/feature_engineering.py`**: Engineer clinically robust dropout-relevant features.
3. **`src/model.py`**: Split/scale data + class balancing utilizing SMOTE.
4. **Training & Comparison**:
   * Logistic Regression (Baseline)
   * Random Forest (Non-linear Ensemble)
   * XGBoost (Boosted Trees)
5. **Evaluation**: Assesses performance via AUC-ROC, F1 Score, Precision, and Recall.

---

## 🔍 Interpretability (SHAP)

Transparency is essential in clinical applications. We utilize **SHAP (SHapley Additive exPlanations)** for robust interpretability:

* **Global Summaries**: Population-level visualizations showing the main risk drivers for treatment dropout.
* **Individual Waterfall Plots**: Patient-level breakdowns indicating which features uniquely contributed to predicting a dropout vs. engagement.

---

## 🛠️ Project Structure

```text
therapy-dropout-prediction/
│
├── api/                   # FastAPI backend for model serving
│   └── main.py
├── app/                   # Streamlit frontend app
│   └── streamlit_app.py
├── data/                  # Directory for synthetic/raw datasets (ignored from git)
├── models/                # Serialized trained model weights (.pkl or .joblib)
├── notebooks/             # Jupyter Notebooks for exploration, training, and interpretability
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_interpretability.ipynb
├── src/                   # Core Python application logic
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   └── model.py
├── venv/                  # Python virtual environment (ignored from git)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 💻 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samartha04/therapy-dropout-prediction.git
   cd therapy-dropout-prediction
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the Notebooks:**
   Explore the data, train models, and run interpretability components in Jupyter.
   ```bash
   jupyter notebook notebooks/
   ```

4. **Serve the API (Development Mode):**
   ```bash
   uvicorn api.main:app --reload
   ```

5. **Start the Streamlit Application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

---

## ⚠️ Ethical Disclaimer

This repository is purely a research prototype intended for educational and methodological exploration.
**It is NOT a medical device, is NOT validated for clinical deployment, and MUST NOT be used as a standalone basis for diagnosis, treatment, or triage decisions.**

---

## 📚 References & Literature

* Swift, J. K., & Greenberg, R. P. (2012). Premature discontinuation in adult psychotherapy.
* McGovern, et al. (2024). Contemporary evidence on psychotherapy dropout predictors.
* Lutz, et al. (2022). Session-level monitoring and psychotherapy outcome/dropout risk factors.
* Gratch, et al. (2014). The Distress Analysis Interview Corpus (DAIC-WOZ).
