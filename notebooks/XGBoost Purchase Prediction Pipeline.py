import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Load Data
# ===============================
df = pd.read_csv("C:\\Users\\ADMIN\\Videos\\smart-shopper-ai\\notebooks\\cleaned_ecommerce_shopper_data.csv")

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Preprocessing Pipeline
# ===============================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# ===============================
# Handle imbalance
# ===============================
positive_ratio = y.mean()
scale_pos_weight = (len(y) - sum(y)) / sum(y)

# ===============================
# Model
# ===============================
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipeline = ImbPipeline(steps=[
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("model", model)
])

# ===============================
# Train
# ===============================
pipeline.fit(X_train, y_train)

# ===============================
# Predict
# ===============================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n✅ TRAIN Accuracy:", accuracy_score(y_train, pipeline.predict(X_train)))
print("✅ TEST Accuracy :", accuracy_score(y_test, y_pred))
print("\n📊 ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import precision_recall_curve

probs = pipeline.predict_proba(X_test)[:, 1]
prec, rec, thresh = precision_recall_curve(y_test, probs)

# Show a few sample thresholds
for p, r, t in zip(prec[:10], rec[:10], thresh[:10]):
    print(f"Threshold={t:.2f} | Precision={p:.2f} | Recall={r:.2f}")


# ===============================
# Confusion Matrix
# ===============================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

