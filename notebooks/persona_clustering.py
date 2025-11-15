# ======================================
# persona_clustering_final_with_personas.py
# ======================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# ======================================
# 1. Load dataset
# ======================================
data = pd.read_csv(r"C:\Users\ADMIN\Videos\Smart clone\Online-Shopper-Buy-or-Bail\notebooks\cleaned_ecommerce_shopper_data.csv")
print("✅ Dataset loaded successfully")
print("Shape:", data.shape)

# ======================================
# 2. Select features for clustering
# ======================================
features = [
    'Administrative_Duration', 'Administrative',
    'ProductRelated_Duration', 'ProductRelated',
    'BounceRates', 'ExitRates', 'PageValues',
    'TotalPages', 'TotalDuration', 'PagesPerSecond',
    'EngagementScore', 'ProductFocus',
    'Revenue', 'ActiveBuyer'
]
X = data[features].copy()

# ======================================
# 3. Feature Weighting Adjustments (Fix A + E)
# ======================================
# Reduce the impact of binary features (Revenue, ActiveBuyer, ReturningVisitorFlag)
'''X['Revenue'] = X['Revenue'] * 0.8
X['ActiveBuyer'] = X['ActiveBuyer'] * 0.5'''


# ======================================
# 4. PCA for visualization and decorrelation
# ======================================
pca = PCA(n_components=9, random_state=42)
X_pca = pca.fit_transform(X)
explained_var = np.sum(pca.explained_variance_ratio_)
print(f"🔹 PCA reduced dimensions from {X.shape[1]} → {X_pca.shape[1]}")
print(f"Explained Variance: {explained_var:.3f}")

# Show explained variance per component
plt.figure(figsize=(7,4))
sns.barplot(x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_, color='steelblue')
plt.title("Explained Variance by PCA Component")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.show()

# Cumulative variance
plt.figure(figsize=(7,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='darkorange')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()

# PCA Component pairplots (first 5)
pca_df = pd.DataFrame(X_pca[:, :5], columns=[f'PC{i+1}' for i in range(5)])
sns.pairplot(pca_df, diag_kind='kde', corner=True)
plt.suptitle("Pairwise PCA Component Distributions (Top 5)", y=1.02)
plt.show()

# ======================================
# 5. Fixed KMeans with k=5
# ======================================
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50, max_iter=1000)
labels = kmeans.fit_predict(X_pca)
data['PersonaCluster'] = labels
silhouette_avg = silhouette_score(X_pca, labels)
print(f"✅ Clustering done with k={best_k}, silhouette score: {silhouette_avg:.3f}")

# ======================================
# 6. Assign Persona Names
# ======================================
persona_labels = {
    0: "Deal Hunter",
    1: "Window Browser",
    2: "Research Shopper",
    3: "Loyal Customer",
    4: "Impulse Buyer"
}
data['Persona'] = data['PersonaCluster'].map(persona_labels)

# ======================================
# 7. Silhouette Plot
# ======================================
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(8, 6)
y_lower = 10
sample_silhouette_values = silhouette_samples(X_pca, labels)
for i in range(best_k):
    ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / best_k)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, persona_labels[i])
    y_lower = y_upper + 10
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Persona Label")
plt.title(f"Silhouette Plot (k={best_k}, score={silhouette_avg:.3f})")
plt.show()

# ======================================
# 8. 2D Visualization of Clusters
# ======================================
pca_2d = PCA(n_components=2, random_state=42)
X_vis = pca_2d.fit_transform(X_pca)
data['PCA1'] = X_vis[:, 0]
data['PCA2'] = X_vis[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Persona', palette='tab10', s=60, alpha=0.85)
centroids_2d = pca_2d.transform(kmeans.cluster_centers_)
plt.scatter(centroids_2d[:,0], centroids_2d[:,1], c='black', s=150, marker='X', label='Centroids')
plt.title(f"Shopper Persona Clusters (k={best_k}, silhouette={silhouette_avg:.3f})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Persona')
plt.grid(True)
plt.show()

# ======================================
# 9. Save labeled clusters
# ======================================
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
output_path = os.path.join(models_dir, "shopper_personas_final_labeled.csv")
data.to_csv(output_path, index=False)
print(f"💾 Clustered dataset with persona labels saved → {output_path}")

# ======================================
# 10. Persona Summary
# ======================================
summary = data.groupby('Persona')[['TotalPages', 'EngagementScore', 'ExitRates', 'Revenue']].mean().round(2)
print("\n🧠 Shopper Persona Summary:")
print(summary)

# ======================================
# 11. Persona Comparison Charts
# ======================================

# --- Bar Chart ---
plt.figure(figsize=(10,6))
summary.plot(kind='bar', figsize=(10,6), colormap='tab10')
plt.title("Persona-wise Comparison of Key Metrics", fontsize=14)
plt.ylabel("Average Value", fontsize=12)
plt.xlabel("Persona", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

# --- Heatmap ---
plt.figure(figsize=(8,5))
sns.heatmap(summary, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Heatmap: Shopper Persona Metric Comparison", fontsize=14)
plt.xlabel("Metrics", fontsize=12)
plt.ylabel("Persona", fontsize=12)
plt.tight_layout()
plt.show()