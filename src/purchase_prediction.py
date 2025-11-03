
"""
Purchase prediction model training script using CatBoostClassifier.

Outputs saved to:
- models/purchase_predictor_catboost.pkl
- outputs/feature_importance.png
- outputs/shap_summary.png
- outputs/confusion_matrix.png
- models/feature_importance.csv
- models/purchase_model_metrics.json

Usage:
    python src/purchase_prediction.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from catboost import CatBoostClassifier, Pool
import shap
import matplotlib.pyplot as plt
import seaborn as sns

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
        raise FileNotFoundError(f"{path} not found. Run data_preprocessing first.")
    return pd.read_csv(path)

print("📂 Loading training data...")
X_train = load_csv("X_train.csv")
X_test = load_csv("X_test.csv")
y_train = load_csv("y_train.csv").values.ravel()
y_test = load_csv("y_test.csv").values.ravel()

feature_names = pickle.load(open(os.path.join(PROCESSED, "feature_names.pkl"), "rb"))

# ---------------- Train Model ----------------
print("🚀 Training CatBoost model...")

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False,
    random_state=42
)

train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

model.fit(train_pool, eval_set=test_pool, verbose=False)

# ---------------- Evaluation ----------------
print("📊 Evaluating model...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
}

print(json.dumps(metrics, indent=2))

# Save metrics
json.dump(metrics, open(os.path.join(MODELS, "purchase_model_metrics.json"), "w"), indent=2)

# ---------------- Confusion Matrix ----------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Purchase Prediction Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "confusion_matrix.png"))
plt.close()

# ---------------- Feature Importance ----------------
importances = model.get_feature_importance()
feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
feat_imp.sort_values(by="importance", ascending=False).to_csv(
    os.path.join(MODELS, "feature_importance.csv"), index=False
)

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=feat_imp.sort_values(by="importance", ascending=False).head(15))
plt.title("Top Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "feature_importance.png"))
plt.close()

# ---------------- SHAP Explainability ----------------
print("🧠 Generating SHAP values (first 300 samples)...")

explainer = shap.TreeExplainer(model)
sample_data = X_test[:300]
shap_values = explainer.shap_values(sample_data)

plt.figure()
shap.summary_plot(shap_values, sample_data, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "shap_summary.png"), dpi=200)
plt.close()

# ---------------- Save Model ----------------
model_path = os.path.join(MODELS, "purchase_predictor_catboost.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model saved to {model_path}")
print("🎉 Purchase prediction training completed successfully!")
