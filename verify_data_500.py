import sys
import os
sys.path.append(os.path.abspath('c:/Users/HP/Desktop/therapy-dropout-prediction/src'))
from data_loader import generate_synthetic_data

df = generate_synthetic_data(n_patients=500, seed=42)
dr = df['dropout'].mean() * 100
corr_attend = df['attendance_consistency'].corr(df['dropout'])
corr_phq9 = df['phq9_score'].corr(df['dropout'])
corr_gap = df['gap_between_sessions_days'].corr(df['dropout'])

with open('c:/Users/HP/Desktop/therapy-dropout-prediction/output_500.txt', 'w') as f:
    f.write(f"Dropout Rate: {dr:.1f}%\n")
    f.write(f"Correlation (Attendance vs Dropout): {corr_attend:.3f}\n")
    f.write(f"Correlation (PHQ9 vs Dropout): {corr_phq9:.3f}\n")
    f.write(f"Correlation (Gap vs Dropout): {corr_gap:.3f}\n")
