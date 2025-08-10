import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

# --- Configuration ---
DATA_PATH = 'data/diabetes_012_health_indicators_BRFSS2015.csv'
MODEL_SAVE_PATH = 'models'
MODEL_FILE_NAME = 'trained_model.pkl'

# Create models directory if not exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# --- 1. Load Data ---
print("🚀 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# --- 2. Validate and Clean Target Column ---
print("🔍 Checking and cleaning target variable...")
if 'Diabetes' not in df.columns:
    raise ValueError("Target column 'Diabetes' not found in dataset!")

# Convert to float for safe handling
df['Diabetes'] = df['Diabetes'].astype(float)

# Keep only valid entries and map: 0→0, 1→1, 2→0 (pre-diabetes treated as non-diabetic)
df = df[df['Diabetes'].isin([0, 1, 2])]
df['Diabetes'] = df['Diabetes'].map({0: 0, 1: 1, 2: 0})  # Map pre-diabetes to 0
df['Diabetes'] = df['Diabetes'].astype(int)

print(f"✅ Cleaned 'Diabetes' column. Unique values: {df['Diabetes'].unique()}")

# --- 3. Features and Target ---
print("📌 Preparing features and target...")
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

# --- 4. Train-Test Split ---
print("SplitOptions...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Preserve class distribution
)

# --- 5. Model Training ---
print("🧠 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
model.fit(X_train, y_train)

# --- 6. Predictions ---
print("📊 Making predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Ensure labels are clean
y_test = y_test.astype(int)

# --- 7. Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)  # Now works: binary 0/1

print("\n✅ Model Evaluation Results")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 8. Save Model ---
model_path = os.path.join(MODEL_SAVE_PATH, MODEL_FILE_NAME)
joblib.dump(model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# --- Feature Importance (Optional) ---
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 10 Important Features:")
print(feature_importance.head(10))