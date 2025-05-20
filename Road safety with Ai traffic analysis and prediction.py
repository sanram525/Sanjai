# main_ai_road_safety.py

# =============================
# Import Required Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import os

# =============================
# Load Dataset
# =============================
df = pd.read_csv("accident_data.csv")

# =============================
# Data Preprocessing
# =============================
# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(axis=1, thresh=0.7 * len(df), inplace=True)

# Feature Engineering
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
df['Hour'] = df['Time'].dt.hour
df['Time_of_Day'] = pd.cut(df['Hour'],
                           bins=[-1, 6, 12, 18, 24],
                           labels=["Night", "Morning", "Afternoon", "Evening"])

df['Is_Weekend'] = pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek >= 5
df['Traffic_Load_Index'] = df['Vehicle_Count'] * df['Accident_Severity']
df['Weather_Score'] = df['Visibility'] / 10

# Drop non-informative or ID columns
drop_cols = ['Accident_ID', 'Date', 'Time', 'Hour']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# =============================
# Encode Categorical Features
# =============================
df = pd.get_dummies(df, drop_first=True)

# =============================
# Split Dataset
# =============================
X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# =============================
# Feature Scaling
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# Train Model (XGBoost)
# =============================
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# =============================
# Model Evaluation
# =============================
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===\n")
print(confusion_matrix(y_test, y_pred))

roc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
print(f"\n=== ROC AUC Score ===\n{roc_score:.4f}")

# =============================
# Plot ROC Curve for Multi-class
# =============================
plt.figure(figsize=(8, 6))
for i in range(y_proba.shape[1]):
    fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
    plt.plot(fpr, tpr, label=f"Class {i}")
plt.title("ROC Curve for Accident Severity Classes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# =============================
# Accident Severity Distribution
# =============================
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Accident Severity Distribution")
plt.xlabel("Severity Class")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("severity_distribution.png")
plt.show()

# =============================
# Save Model and Scaler
# =============================
joblib.dump(model, "xgboost_accident_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# =============================
# Optional: Risk Zone Mapping
# =============================
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    accident_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3,
            color='red' if row['Accident_Severity'] > 1 else 'green',
            fill=True
        ).add_to(accident_map)
    map_path = "accident_hotspots.html"
    accident_map.save(map_path)
    print(f"\n✅ Accident hotspot map saved to: {os.path.abspath(map_path)}")
