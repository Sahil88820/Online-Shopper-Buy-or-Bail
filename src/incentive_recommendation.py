"""
Incentive recommendation model training using XGBoost multi-class classifier.

Outputs:
- models/incentive_recommender.pkl
- models/incentive_label_encoder.pkl
- outputs/incentive_confusion_matrix.png
- outputs/incentive_feature_importance.png
- models/incentive_model_metrics.json

Usage:
    python src/incentive_recommendation.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROCESSED = os.path.join(BASE, "data", "processed")
MODELS = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")

os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# ---------------- Load Data ----------------
def load_csv(name):
    path = os.path.join(PROCESSED, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run previous steps first.")
    return pd.read_csv(path)

print("📂 Loading training data...")

X_train = load_csv("X_train.csv")
X_test = load_csv("X_test.csv")

# Simulated incentive label generation (until real logs available)
# You can replace this later with real incentive acceptance logs
def generate_incentive_labels(n):
    classes = [
        "discount_10", "discount_15", "discount_20",
        "loyalty_points", "free_shipping", "urgency_banner", "none"
    ]
    # Probability distribution (can tweak)
    probs = [0.15, 0.15, 0.2, 0.15, 0.15, 0.1, 0.1]
    return np.random.choice(classes, size=n, p=probs)

# Load y if exists, else generate temporary synthetic labels
y_train_path = os.path.join(PROCESSED, "y_train_incentive.csv")
y_test_path = os.path.join(PROCESSED, "y_test_incentive.csv")

if os.path.exists(y_train_path):
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()
    print("✅ Loaded real incentive labels")
else:
    print("⚠️ No incentive labels found — generating synthetic ones.")
    y_train = generate_incentive_labels(len(X_train))
    y_test = generate_incentive_labels(len(X_test))
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

# Encode labels
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# ---------------- Train Model ----------------
print("🚀 Training XGBoost model for incentives...")

model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(label_encoder.classes_),
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train_enc)

# ---------------- Evaluate ----------------
print("📊 Evaluating model...")

y_pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test_enc, y_pred)),
    "classification_report": classification_report(y_test_enc, y_pred, target_names=label_encoder.classes_)
}

print(json.dumps({"accuracy": metrics["accuracy"]}, indent=2))

# Save metrics
json.dump(metrics, open(os.path.join(MODELS, "incentive_model_metrics.json"), "w"), indent=2)

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Incentive Recommendation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "incentive_confusion_matrix.png"))
plt.close()

# ---------------- Feature Importance ----------------
importance = model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)

feat_imp.to_csv(os.path.join(MODELS, "incentive_feature_importance.csv"), index=False)

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=feat_imp.head(15))
plt.title("Incentive Model Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "incentive_feature_importance.png"))
plt.close()

# ---------------- Save Model ----------------
pickle.dump(model, open(os.path.join(MODELS, "incentive_recommender.pkl"), "wb"))
pickle.dump(label_encoder, open(os.path.join(MODELS, "incentive_label_encoder.pkl"), "wb"))

print("✅ Incentive model saved!")
print("🎉 Incentive recommendation training complete.")
