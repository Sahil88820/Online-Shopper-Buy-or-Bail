"""
Persona clustering script.

- Loads processed features from data/processed/X_train.csv (expects scaled features).
- Runs KMeans clustering (default k=5, configurable).
- Computes silhouette score and inertia for k range and saves elbow + silhouette plots.
- Fits PCA (n_components=2) for 2D visualization.
- Produces simple persona profiles (cluster means) and saves them.
- Saves models:
    - models/kmeans_model.pkl
    - models/pca_transformer.pkl
    - models/persona_profiles.pkl

Usage:
    python src/persona_clustering.py
    python src/persona_clustering.py --k 5
    python src/persona_clustering.py --k 5 --kmax 8
"""

import os
import argparse
import pickle
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# ---------- Paths ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------- Helper functions ----------
def load_features(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed features not found at {path}. Run data_preprocessing first.")
    df = pd.read_csv(path)
    return df

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------- Main pipeline ----------
def run_clustering(
    x_train_path: str,
    k: int = 5,
    kmax: int = 8,
    random_state: int = 42,
):
    print("Loading processed features...")
    X_train = load_features(x_train_path)
    feature_names = list(X_train.columns)
    X = X_train.values

    # Try a range of k to produce elbow + silhouette diagnostics
    k_range = list(range(2, kmax + 1))
    inertias = []
    silhouettes = []

    print(f"Running KMeans diagnostics for k in {k_range} ...")
    for kk in k_range:
        km = KMeans(n_clusters=kk, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = float("nan")
        silhouettes.append(sil)
        print(f" k={kk:2d}  inertia={km.inertia_:.2f}  silhouette={sil:.4f}")

    # Save elbow (inertia) + silhouette plots
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker="o")
    plt.title("Elbow curve (Inertia)")
    plt.xlabel("n_clusters")
    plt.ylabel("Inertia")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, marker="o")
    plt.title("Silhouette scores")
    plt.xlabel("n_clusters")
    plt.ylabel("Silhouette score")
    plt.grid(True)

    elbow_path = os.path.join(OUTPUTS_DIR, "cluster_optimization.png")
    plt.tight_layout()
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"Saved cluster diagnostics to: {elbow_path}")

    # Fit final KMeans with chosen k
    print(f"Fitting final KMeans with k={k} ...")
    final_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=50)
    cluster_labels = final_kmeans.fit_predict(X)

    # PCA for 2D visualization
    print("Fitting PCA (n_components=2) for visualization ...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # Create 2D scatter plot
    plt.figure(figsize=(7, 6))
    for cluster_id in range(k):
        idx = cluster_labels == cluster_id
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"cluster {cluster_id}", alpha=0.6, s=12)
    plt.legend()
    plt.title(f"Persona clusters (k={k}) - PCA projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    persona_plot_path = os.path.join(OUTPUTS_DIR, "persona_clusters.png")
    plt.tight_layout()
    plt.savefig(persona_plot_path, dpi=150)
    plt.close()
    print(f"Saved persona 2D visualization to: {persona_plot_path}")

    # Build persona profiles (cluster-wise means + size)
    print("Building persona profiles ...")
    df_X = pd.DataFrame(X, columns=feature_names)
    df_X["cluster"] = cluster_labels

    persona_profiles: Dict[int, Dict[str, Any]] = {}
    for cid in sorted(df_X["cluster"].unique()):
        sub = df_X[df_X["cluster"] == cid].drop(columns=["cluster"])
        means = sub.mean().to_dict()
        medians = sub.median().to_dict()
        size = int(sub.shape[0])
        persona_profiles[int(cid)] = {
            "cluster_id": int(cid),
            "size": size,
            "percentage": float(size) / float(df_X.shape[0]),
            "means": means,
            "medians": medians,
        }

    # Heuristic-based persona naming (optional, best-effort)
    # We try to inspect a few key features if present in the data and assign friendly names.
    # If your feature set is different, edit this mapping logic.
    def pick_persona_name(means: Dict[str, float]) -> str:
        # Safe default
        name = "Cluster"
        # heuristics: look for common feature names found in the README/spec
        # (ProductRelated, TotalDuration, TotalPages, EngagementScore, PageValues)
        pr = means.get("ProductRelated", 0) + means.get("Product_Related", 0) + means.get("product_related", 0)
        dur = means.get("TotalDuration", 0)
        pages = means.get("TotalPages", 0) + means.get("Total_Pages", 0)
        engagement = means.get("EngagementScore", 0) + means.get("engagement_score", 0)
        page_values = means.get("PageValues", means.get("Page_Values", 0))

        # Simple rules
        if pr > 1.5 and page_values > 0.5:
            name = "Deal Hunter"
        elif dur < 100 and pr > 1.0:
            name = "Impulse Buyer"
        elif engagement < 50 and pr < 1.0:
            name = "Window Browser"
        elif dur > 400 and pages > 20:
            name = "Research Shopper"
        elif pages > 10 and page_values > 10:
            name = "Loyal Customer"
        else:
            name = "Mixed Shopper"
        return name

    # Attach friendly names
    for cid, profile in persona_profiles.items():
        profile_name = pick_persona_name(profile["means"])
        profile["persona_type"] = profile_name

    # Save models + profiles
    kmeans_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    pca_path = os.path.join(MODELS_DIR, "pca_transformer.pkl")
    profiles_path = os.path.join(MODELS_DIR, "persona_profiles.pkl")

    save_pickle(final_kmeans, kmeans_path)
    save_pickle(pca, pca_path)
    save_pickle(persona_profiles, profiles_path)

    print(f"Saved KMeans model to: {kmeans_path}")
    print(f"Saved PCA transformer to: {pca_path}")
    print(f"Saved persona profiles to: {profiles_path}")

    # Small JSON summary for API/analytics consumption
    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_clusters": int(k),
        "cluster_sizes": {str(cid): persona_profiles[cid]["size"] for cid in persona_profiles},
    }
    summary_path = os.path.join(MODELS_DIR, "persona_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved persona summary JSON to: {summary_path}")

    return {
        "kmeans_path": kmeans_path,
        "pca_path": pca_path,
        "profiles_path": profiles_path,
        "persona_plot": persona_plot_path,
        "elbow_plot": elbow_path,
    }

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persona clustering pipeline")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters to fit (default: 5)")
    parser.add_argument("--kmax", type=int, default=8, help="Max k to evaluate for diagnostics (default: 8)")
    parser.add_argument("--xtrain", type=str, default=os.path.join(PROCESSED_DIR, "X_train.csv"),
                        help="Path to processed X_train.csv (default: data/processed/X_train.csv)")
    args = parser.parse_args()

    results = run_clustering(x_train_path=args.xtrain, k=args.k, kmax=args.kmax)
    print("Done. Artifacts:")
    for k, v in results.items():
        print(f" - {k}: {v}")
