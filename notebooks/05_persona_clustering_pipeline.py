import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score
import hdbscan
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def perform_persona_clustering_hdbscan():
    print("📂 Loading dataset...")
    df = pd.read_csv("C:\\Users\\ADMIN\\Videos\\smart-shopper-ai\\notebooks\\cleaned_ecommerce_shopper_data_standardscaler.csv")

    # Ensure target column exists
    if "Revenue" not in df.columns:
        raise ValueError("❌ 'Revenue' column not found in dataset!")

    # ✅ Feature engineering — create behavior ratios
    print("🧮 Creating derived behavioral features...")
    df["Admin_Efficiency"] = df["Administrative_Duration"] / (df["Administrative"] + 1)
    df["Info_Efficiency"] = df["Informational_Duration"] / (df["Informational"] + 1)
    df["Product_Efficiency"] = df["ProductRelated_Duration"] / (df["ProductRelated"] + 1)
    df["ExitBounceRatio"] = df["ExitRates"] / (df["BounceRates"] + 1e-5)

    # Separate features and target
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    # ✅ Robust scaling
    print("⚖️ Scaling data with RobustScaler (outlier-resistant)...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Data scaled successfully.")

    # ✅ Dimensionality reduction
    print("🔻 Applying PCA (retain 85% variance)...")
    pca = PCA(n_components=0.85, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Ensure at least 2D for visualization
    if X_pca.shape[1] < 2:
        print(f"⚠️ Only {X_pca.shape[1]} PCA component found. Forcing 2D PCA for visualization.")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
    print(f"✅ PCA reduced features to {X_pca.shape[1]} components explaining ~85% variance.")

    # ✅ HDBSCAN clustering
    print("\n🚀 Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=5,
        metric='euclidean',
        cluster_selection_epsilon=0.1
    )
    cluster_labels = clusterer.fit_predict(X_pca)

    df["PersonaCluster"] = cluster_labels
    n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))
    noise_ratio = np.mean(cluster_labels == -1) * 100
    print(f"✅ HDBSCAN found {n_clusters} clusters (+ {noise_ratio:.2f}% noise).")

    # ✅ Silhouette (non-noise)
    mask = cluster_labels != -1
    if n_clusters > 1:
        sil_score = silhouette_score(X_pca[mask], cluster_labels[mask])
        print(f"✅ Silhouette Score (non-noise): {sil_score:.4f}")
    else:
        sil_score = np.nan
        print("⚠️ Only one cluster found — silhouette score not applicable.")

    # ✅ Cluster summary
    print("\n📊 Cluster Summary (Conversion Rate per Persona):")
    cluster_summary = (
        df[df["PersonaCluster"] != -1]
        .groupby("PersonaCluster")["Revenue"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "TotalUsers", "sum": "Buyers", "mean": "ConversionRate"})
        .sort_values("ConversionRate", ascending=False)
    )
    print(cluster_summary)

    # ✅ Persona naming
    persona_map = {}
    sorted_clusters = cluster_summary.reset_index().sort_values("ConversionRate", ascending=False)
    for i, row in enumerate(sorted_clusters.itertuples()):
        if i == 0:
            persona_map[row.PersonaCluster] = "Decisive Buyer"
        elif i == 1:
            persona_map[row.PersonaCluster] = "Engaged Browser"
        elif i == 2:
            persona_map[row.PersonaCluster] = "Window Shopper"
        else:
            persona_map[row.PersonaCluster] = f"Segment_{i+1}"

    df["PersonaType"] = df["PersonaCluster"].map(persona_map)
    df["PersonaType"] = df["PersonaType"].fillna("Noise")

    # ✅ Confusion matrix (for reference only)
    non_noise = df[df["PersonaCluster"] != -1]
    cluster_majority = non_noise.groupby("PersonaCluster")["Revenue"].agg(lambda x: 1 if x.mean() >= 0.5 else 0).to_dict()
    predicted_labels = non_noise["PersonaCluster"].map(cluster_majority)
    acc = accuracy_score(non_noise["Revenue"], predicted_labels)
    cm = confusion_matrix(non_noise["Revenue"], predicted_labels)
    print(f"\n✅ Cluster-Label Alignment Accuracy: {acc:.4f}")
    print("📄 Confusion Matrix:\n", cm)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
    plt.title("Cluster vs. Actual Revenue Alignment")
    plt.xlabel("Predicted (Cluster → Revenue)")
    plt.ylabel("Actual Revenue")
    plt.show()

    # ✅ Visualization (safe)
    if X_pca.shape[1] >= 2:
        plt.figure(figsize=(8,6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=10, alpha=0.7)
        plt.title("HDBSCAN Clusters in PCA Space")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()
    else:
        print("⚠️ PCA has <2 components, skipping 2D cluster plot.")

    # ✅ Save models and results
    joblib.dump(scaler, "persona_scaler.pkl")
    joblib.dump(pca, "persona_pca.pkl")
    joblib.dump(clusterer, "persona_hdbscan.pkl")
    df.to_csv("shopper_personas_hdbscan.csv", index=False)

    print("\n💾 Saved:")
    print(" - persona_scaler.pkl")
    print(" - persona_pca.pkl")
    print(" - persona_hdbscan.pkl")
    print(" - shopper_personas_hdbscan.csv")
    print("\n✅ Persona clustering with HDBSCAN complete!")

if __name__ == "__main__":
    perform_persona_clustering_hdbscan()
