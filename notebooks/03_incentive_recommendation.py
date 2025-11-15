# train_incentive_model.py
"""
Model 3: Incentive Recommendation (multi-class)
Usage:
    python train_incentive_model.py

Requirements:
    pip install pandas numpy scikit-learn xgboost joblib imbalanced-learn
Files expected (in same folder or provide full paths):
    - cleaned_ecommerce_shopper_data.csv       (required)
    - shopper_personas.csv                     (optional; fallback to persona pipeline)
    - persona_scaler.pkl, persona_pca.pkl, persona_kmeans.pkl (optional)
    - xgboost_pipeline.pkl                     (optional; contains purchase predictor pipeline)
Outputs:
    - incentive_pipeline.pkl
    - incentive_label_encoder.pkl
    - incentive_train_report.txt
    - incentive_test_predictions_first1000.csv
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ---------------------------
# Config / mappings
# ---------------------------
DATA_CLEANED = r"C:\Users\ADMIN\Videos\Smart clone\Online-Shopper-Buy-or-Bail\notebooks\cleaned_ecommerce_shopper_data.csv"
PERSONA_CSV = "shopper_personas.csv"

PERSONA_PIPE_FILES = ("persona_scaler.pkl", "persona_pca.pkl", "persona_kmeans.pkl")
PURCHASE_PIPE = "xgboost_pipeline.pkl"  # used to get purchase probability if available

OUT_MODEL = "incentive_pipeline.pkl"
OUT_ENCODER = "incentive_label_encoder.pkl"
OUT_REPORT = "incentive_train_report.txt"
OUT_FIRST1000 = "incentive_test_predictions_first1000.csv"

# persona -> recommended incentive mapping (finalized)
PERSONA_TO_INCENTIVE = {
    "Deal Hunter": "bundle_offer",
    "deal_hunter": "bundle_offer",
    "Loyal Customer": "free_shipping",
    "loyal_customer": "free_shipping",
    "Research Shopper": "discount_10",
    "research_shopper": "discount_10",
    "Impulse Buyer": "urgency_banner",
    "impulse_buyer": "urgency_banner",
    "Window Shopper": "social_proof_banner",
    "window_shopper": "social_proof_banner",
}

RANDOM_STATE = 42

# ---------------------------
# Helpers
# ---------------------------
def load_cleaned_data(path=DATA_CLEANED):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned dataset not found at {path}")
    df = pd.read_csv(path)
    return df

def attach_persona(df):
    """
    Try to attach persona type to df. First look for shopper_personas.csv.
    If not present, try to load persona pipeline (scaler+pca+kmeans) to compute clusters,
    and map most common cluster -> PersonaType if shopper_personas.csv not available.
    """
    if os.path.exists(PERSONA_CSV):
        print(f"ℹ️ Loading persona file: {PERSONA_CSV}")
        p = pd.read_csv(PERSONA_CSV)
        # Expect there to be columns like PersonaCluster and PersonaType and an identifier to merge.
        # We'll try a safe merge: if p and df share index length, simply append the PersonaType column.
        # Otherwise try to merge on common columns.
        if len(p) == len(df):
            df = df.copy()
            if "PersonaType" in p.columns:
                df["PersonaType"] = p["PersonaType"].values
                print("✅ PersonaType attached from shopper_personas.csv")
                return df
            elif "PersonaCluster" in p.columns:
                df["PersonaCluster"] = p["PersonaCluster"].values
                print("✅ PersonaCluster attached from shopper_personas.csv (no PersonaType column)")
                # PersonaType may not exist; can map later
                return df
            else:
                print("⚠️ shopper_personas.csv found but missing PersonaType/PersonaCluster columns. Falling back.")
        else:
            print("⚠️ shopper_personas.csv size mismatch; falling back to persona pipeline.")
    # fallback: compute persona via persona pipeline
    scaler_file, pca_file, kmeans_file = PERSONA_PIPE_FILES
    if all(os.path.exists(f) for f in PERSONA_PIPE_FILES):
        print("ℹ️ Persona files found, computing persona clusters...")
        scaler = joblib.load(scaler_file)
        pca = joblib.load(pca_file)
        kmeans = joblib.load(kmeans_file)
        # we expect persona scaler expects original feature columns; attempt to use df.drop('Revenue')
        X = df.drop(columns=[c for c in ["Revenue"] if c in df.columns], errors='ignore')
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        clusters = kmeans.predict(X_pca)
        df = df.copy()
        df["PersonaCluster"] = clusters
        print("✅ PersonaCluster computed via persona pipeline")
        return df
    else:
        print("⚠️ No persona information available (no shopper_personas.csv and no persona pipeline files).")
        return df


def add_purchase_probability(df):
    """
    Use existing purchase model pipeline to add purchase probability feature.
    If purchase pipeline not found, we skip this step.
    """
    if os.path.exists(PURCHASE_PIPE):
        print(f"ℹ️ Loading purchase pipeline: {PURCHASE_PIPE}")
        pipeline = joblib.load(PURCHASE_PIPE)
        # pipeline expects feature columns used during training.
        X = df.drop(columns=[c for c in ["Revenue"] if c in df.columns], errors='ignore')
        try:
            probs = pipeline.predict_proba(X)[:, 1]
            df = df.copy()
            df["purchase_prob"] = probs
            print("✅ purchase_prob added from purchase pipeline")
        except Exception as e:
            print("⚠️ Failed to compute purchase_prob using pipeline:", e)
            print("Proceeding without purchase_prob feature.")
    else:
        print("⚠️ Purchase pipeline not found; continuing without purchase_prob.")
    return df

def map_persona_to_incentive(df):
    """
    Map persona to incentive. This function first looks for a PersonaType text field.
    If not present but PersonaCluster exists, we try to create PersonaType by cluster->most common persona mapping
    (this is a fallback and requires manual inspection ideally).
    """
    df = df.copy()
    if "PersonaType" not in df.columns and "PersonaCluster" in df.columns:
        # Try to map cluster->PersonaType by heuristics (not ideal). We'll map cluster with highest conversion rate to "Deal Hunter", etc.
        print("ℹ️ PersonaType not present. Creating PersonaType from PersonaCluster via conversion-rate heuristic.")
        cluster_summary = (
            df.groupby("PersonaCluster")["Revenue"]
            .agg(["count", "sum", "mean"])
            .rename(columns={"count": "TotalUsers", "sum": "Buyers", "mean": "ConversionRate"})
            .sort_values("ConversionRate", ascending=False)
        ).reset_index()
        # assign persona names by descending conversion rate (heuristic)
        persona_labels = ["Deal Hunter", "Loyal Customer", "Research Shopper", "Impulse Buyer", "Window Shopper"]
        cluster_to_persona = {}
        for idx, row in enumerate(cluster_summary.itertuples()):
            name = persona_labels[idx] if idx < len(persona_labels) else f"Segment_{idx+1}"
            cluster_to_persona[row.PersonaCluster] = name
        df["PersonaType"] = df["PersonaCluster"].map(cluster_to_persona)
        print("✅ PersonaType heuristically assigned from PersonaCluster. Inspect shopper_personas.csv for better control.")
    elif "PersonaType" in df.columns:
        print("✅ PersonaType column already present.")
    else:
        print("⚠️ No persona columns present. You must supply persona info for label generation.")
        raise RuntimeError("Persona information missing; cannot map incentives reliably.")
    # Map persona to incentive label
    df["incentive_label"] = df["PersonaType"].map(PERSONA_TO_INCENTIVE)
    # if mapping produced NaNs, fill with a safe default
    df["incentive_label"] = df["incentive_label"].fillna("social_proof_banner")
    print("✅ incentive_label column created.")
    return df

# ---------------------------
# Training pipeline
# ---------------------------
def train_incentive_model(df, save_model=True):
    # prepare X, y
    features = df.drop(columns=[c for c in ["Revenue", "PersonaType", "PersonaCluster", "incentive_label"] if c in df.columns], errors='ignore')
    y = df["incentive_label"]
    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"ℹ️ Found {num_classes} incentive classes: {list(le.classes_)}")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    # imbalance handling: SMOTE for multiclass (may not always be best, but good baseline)
    try:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print("✅ SMOTE applied to training set.")
    except Exception as e:
        print("⚠️ SMOTE failed or not applicable:", e)
        X_train_res, y_train_res = X_train, y_train

    # build pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    # quick hyperparam grid for GridSearch (optional; you can skip for speed)
    param_grid = {
        "xgb__n_estimators": [150, 350],
        "xgb__max_depth": [4, 6],
        "xgb__learning_rate": [0.05, 0.1]
    }

    print("🚀 Starting GridSearchCV (may take time)...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1_macro", verbose=0, n_jobs=-1)
    grid.fit(X_train_res, y_train_res)

    best = grid.best_estimator_
    print("✅ Best params:", grid.best_params_)

    # Evaluate
    y_train_pred = best.predict(X_train)
    y_test_pred = best.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    f1_macro = f1_score(y_test, y_test_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    cls_report = classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    # Save model and encoder
    if save_model:
        joblib.dump(best, OUT_MODEL)
        joblib.dump(le, OUT_ENCODER)
        print(f"💾 Saved model -> {OUT_MODEL}")
        print(f"💾 Saved label encoder -> {OUT_ENCODER}")

    # Save report
    with open(OUT_REPORT, "w") as f:
        f.write("Incentive Recommendation Model Report\n")
        f.write("===============================\n\n")
        f.write(f"Best params: {grid.best_params_}\n\n")
        f.write(f"Train Accuracy: {acc_train:.4f}\n")
        f.write(f"Test Accuracy : {acc_test:.4f}\n")
        f.write(f"Balanced Acc : {bal_acc:.4f}\n")
        f.write(f"F1 Macro      : {f1_macro:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(cls_report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")
    print(f"💾 Training report saved to {OUT_REPORT}")

    print("\n📊 Summary metrics:")
    print(f"Train Accuracy: {acc_train:.4f}")
    print(f"Test Accuracy : {acc_test:.4f}")
    print(f"Balanced Acc  : {bal_acc:.4f}")
    print(f"F1 Macro      : {f1_macro:.4f}")
    print("\nClassification Report:\n", cls_report)
    print("\nConfusion Matrix:\n", cm)

    return best, le

# ---------------------------
# Cross-check first 1000 rows
# ---------------------------
def cross_check_first_1000(pipeline, label_encoder, df):
    print("\n🔎 Cross-checking first 1000 rows from cleaned dataset")
    df_small = df.head(1000).copy()
    X_small = df_small.drop(columns=[c for c in ["Revenue", "PersonaType", "PersonaCluster", "incentive_label"] if c in df_small.columns], errors='ignore')
    y_true_labels = df_small["incentive_label"].values if "incentive_label" in df_small.columns else None

    probs = pipeline.predict_proba(X_small)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)

    results_df = pd.DataFrame({
        "index": df_small.index,
        "true_incentive": y_true_labels,
        "pred_incentive": pred_labels,
        "pred_prob": np.max(probs, axis=1)
    })
    # compute correctness only if true labels exist
    if y_true_labels is not None:
        correct = (results_df["true_incentive"] == results_df["pred_incentive"]).sum()
        total = len(results_df)
        acc = correct / total * 100
        print(f"✅ First 1000 cross-check accuracy: {acc:.2f}% ({correct}/{total})")
    else:
        print("⚠️ True incentive labels not available for first 1000 rows; predictions still saved.")

    results_df.to_csv(OUT_FIRST1000, index=False)
    print(f"💾 Saved cross-check results: {OUT_FIRST1000}")
    return results_df

# ---------------------------
# Inference helper
# ---------------------------
def predict_incentive(pipeline, label_encoder, feature_df):
    """
    Given a feature dataframe (single row or many), return predicted incentive label and probability.
    feature_df: pandas DataFrame with same features used during training.
    """
    probs = pipeline.predict_proba(feature_df)
    idx = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(idx)
    confs = probs[np.arange(len(idx)), idx]
    return labels, confs

# ---------------------------
# Main flow
# ---------------------------
def main():
    print("📂 Loading cleaned dataset...")
    df = load_cleaned_data(DATA_CLEANED)

    print("🔗 Attaching persona information...")
    df = attach_persona(df)

    print("📈 Adding purchase probability (optional)...")
    df = add_purchase_probability(df)

    print("🔖 Creating incentive labels from persona mapping...")
    df = map_persona_to_incentive(df)

    print("🧪 Starting model training for incentive recommendation...")
    model_pipeline, label_encoder = train_incentive_model(df, save_model=True)

    print("🧾 Cross-checking first 1000 rows...")
    cross_results = cross_check_first_1000(model_pipeline, label_encoder, df)

    print("\n✅ All done. You can use `predict_incentive()` by loading")
    print(f"   model pipeline: joblib.load('{OUT_MODEL}')")
    print(f"   label encoder : joblib.load('{OUT_ENCODER}')")
    print("Example usage:")
    print(">>> pipeline = joblib.load('incentive_pipeline.pkl')")
    print(">>> le = joblib.load('incentive_label_encoder.pkl')")
    print(">>> labels, confs = predict_incentive(pipeline, le, X_to_predict)")

if __name__ == "__main__":
    main()
