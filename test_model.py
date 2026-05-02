import pickle
import pandas as pd

model = pickle.load(open('models/xgboost_model.pkl', 'rb'))
from src.evaluate import compute_risk_score

low = pd.DataFrame([{
    'phq9_score': 3,
    'session_number': 18,
    'session_frequency_per_month': 7.0,
    'attendance_consistency': 0.95,
    'gap_between_sessions_days': 4,
    'mood_rating': 9,
    'age': 30,
    'phq9_change_rate': -1.8,
    'gap_increasing': 0,
    'max_attendance_streak': 17,
    'phq9_change_rate_abs': 1.8
}])

high = pd.DataFrame([{
    'phq9_score': 24,
    'session_number': 2,
    'session_frequency_per_month': 1.0,
    'attendance_consistency': 0.10,
    'gap_between_sessions_days': 55,
    'mood_rating': 2,
    'age': 30,
    'phq9_change_rate': 1.8,
    'gap_increasing': 1,
    'max_attendance_streak': 0,
    'phq9_change_rate_abs': 1.8
}])

moderate = pd.DataFrame([{
    'phq9_score': 16,
    'session_number': 8,
    'session_frequency_per_month': 3.0,
    'attendance_consistency': 0.35,
    'gap_between_sessions_days': 38,
    'mood_rating': 4,
    'age': 30,
    'phq9_change_rate': 0.3,
    'gap_increasing': 1,
    'max_attendance_streak': 3,
    'phq9_change_rate_abs': 0.3
}])

print("LOW RISK:", compute_risk_score(model, low))
print("HIGH RISK:", compute_risk_score(model, high))
print("MODERATE:", compute_risk_score(model, moderate))