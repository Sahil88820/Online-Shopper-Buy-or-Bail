import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
#  TRAIN & EVALUATE MODEL
# =========================
def train_and_evaluate():
    print("📂 Loading cleaned dataset...")
    df = pd.read_csv("C:\\Users\\ADMIN\\Videos\\smart-shopper-ai\\notebooks\\cleaned_ecommerce_shopper_data.csv")

    # Features & target
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance
    neg, pos = np.sum(y_train == 0), np.sum(y_train == 1)
    scale_pos_weight = neg / pos
    print(f"⚖️ scale_pos_weight = {scale_pos_weight:.2f}")

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # Pipeline: Scale + Model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=6,
            learning_rate=0.08,
            n_estimators=450,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            eval_metric="logloss",
            random_state=42
        ))
    ])

    print("🚀 Training XGBoost model...")
    pipeline.fit(X_train_sm, y_train_sm)

    # Save model
    joblib.dump(pipeline, "xgboost_pipeline.pkl")
    print("✅ Model saved as xgboost_pipeline.pkl\n")

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1]

    print(f"✅ Train Accuracy: {accuracy_score(y_train, pipeline.predict(X_train)):.4f}")
    print(f"✅ Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"🎯 ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n")

    print("📄 Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.show()

    # Threshold Optimization
    print("🎚️ Optimizing threshold...")
    best_t = 0
    best_f1 = 0
    for t in np.arange(0.1, 1, 0.01):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1 = score
            best_t = t

    joblib.dump(best_t, "xgb_best_threshold.pkl")
    print(f"✅ Best Threshold saved: {best_t:.2f} (F1={best_f1:.4f})\n")

    return pipeline, best_t


# =========================================
# VALIDATE ON FULL CLEANED DATASET
# =========================================
def validate_on_cleaned_data():
    print("📂 Validating full cleaned dataset...")
    df = pd.read_csv("C:\\Users\\ADMIN\\Videos\\smart-shopper-ai\\notebooks\\cleaned_ecommerce_shopper_data.csv")
    X = df.drop("Revenue", axis=1)
    y_true = df["Revenue"]

    pipe = joblib.load("xgboost_pipeline.pkl")
    threshold = joblib.load("xgb_best_threshold.pkl")

    y_prob = pipe.predict_proba(X)[:,1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = (y_pred == y_true).mean() * 100
    print(f"\n✅ Full Cleaned Dataset Accuracy: {accuracy:.2f}%")
    print(f"❌ Incorrect: {100 - accuracy:.2f}%")

    results = pd.DataFrame({
        "True": y_true.values,
        "Pred": y_pred,
        "Prob": np.round(y_prob, 4)
    })

    results.to_csv("validation_cleaned_results.csv", index=False)
    print("💾 Saved: validation_cleaned_results.csv")
    print("\n📋 First 20 rows:")
    print(results.head(20))


# =========================================
# CROSS-CHECK FIRST 1000 CLEANED ROWS
# =========================================
def cross_check_cleaned_first_1000():
    print("\n📂 Cross-checking first 1000 cleaned rows...")
    df = pd.read_csv("C:\\Users\\ADMIN\\Videos\\smart-shopper-ai\\notebooks\\cleaned_ecommerce_shopper_data.csv").head(1000)

    y_true = df["Revenue"]
    X = df.drop("Revenue", axis=1)

    pipe = joblib.load("xgboost_pipeline.pkl")
    threshold = joblib.load("xgb_best_threshold.pkl")

    y_prob = pipe.predict_proba(X)[:,1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = (y_pred == y_true).mean() * 100
    print(f"✅ Accuracy (first 1000 rows): {accuracy:.2f}%")
    print(f"❌ Incorrect: {100 - accuracy:.2f}%")

    results = pd.DataFrame({
        "Index": df.index,
        "True": y_true.values,
        "Pred": y_pred,
        "Prob": np.round(y_prob, 4)
    })

    results.to_csv("crosscheck_cleaned_first_1000.csv", index=False)
    print("💾 Saved: crosscheck_cleaned_first_1000.csv")

    print("\n📋 First 20 rows:")
    print(results.head(20))


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    model, thr = train_and_evaluate()
    validate_on_cleaned_data()
    cross_check_cleaned_first_1000()
