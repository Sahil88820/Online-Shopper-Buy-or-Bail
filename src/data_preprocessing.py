import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# =====================================
# Paths
# =====================================
RAW_DATA_PATH = "data/raw/online_shoppers_intention.csv"
PROCESSED_DIR = "data/processed/"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv(RAW_DATA_PATH)

# =====================================
# Basic Cleaning
# =====================================
df.drop_duplicates(inplace=True)

# Convert boolean-like to true booleans
df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# =====================================
# Feature Engineering
# =====================================
df["TotalDuration"] = (
    df["Administrative_Duration"] +
    df["Informational_Duration"] +
    df["ProductRelated_Duration"]
)

df["TotalPages"] = (
    df["Administrative"] +
    df["Informational"] +
    df["ProductRelated"]
)

df["EngagementScore"] = df["TotalDuration"] * (1 - df["BounceRates"])

df["ProductFocusRatio"] = df["ProductRelated"] / (df["TotalPages"] + 1)
df["SessionIntensity"] = df["TotalDuration"] / (df["TotalPages"] + 1)

df["ExitFlag"] = (df["ExitRates"] > 0.1).astype(int)
df["BounceFlag"] = (df["BounceRates"] > 0.1).astype(int)

# Fill any NA generated
df.fillna(0, inplace=True)

# =====================================
# Encode Categorical
# =====================================
label_encoders = {}
categorical_cols = ["Month", "VisitorType"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =====================================
# Train/Test Split
# =====================================
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================
# Scale
# =====================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# Save Outputs
# =====================================
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(PROCESSED_DIR + "X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(PROCESSED_DIR + "X_test.csv", index=False)
pd.DataFrame(y_train).to_csv(PROCESSED_DIR + "y_train.csv", index=False)
pd.DataFrame(y_test).to_csv(PROCESSED_DIR + "y_test.csv", index=False)

pickle.dump(scaler, open(PROCESSED_DIR + "scaler.pkl", "wb"))
pickle.dump(label_encoders, open(PROCESSED_DIR + "label_encoders.pkl", "wb"))
pickle.dump(list(X.columns), open(PROCESSED_DIR + "feature_names.pkl", "wb"))

print("✅ Data preprocessing complete!")
print(f"Processed files saved to {PROCESSED_DIR}")
