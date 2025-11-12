# persona_clustering_optimized_with_visuals.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# ======================================
# 1. Load dataset
# ======================================
data = pd.read_csv(r"C:\Users\aagra\Desktop\Online shopper\notebooks\cleaned_ecommerce_shopper_data.csv")
print("✅ Dataset loaded successfully")
print("Shape:", data.shape)

# ======================================
# 2. Select clustering features
# ======================================
features = [
    'BounceRates', 'ExitRates', 'HighBounce','HighExit', 'ActiveBuyer',
    'TotalPages', 'TotalDuration', 'PagesPerSecond',
    'EngagementScore', 'ProductFocus', 'ReturningVisitorFlag',
    'ProductEngagement', 'InfoEfficiency', 'AdminEfficiency'
]
X = data[features].copy()

# ======================================
# 3. Scale data (important for distance-based clustering)
# ======================================
'''scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)'''

# ======================================
# 4. PCA (decorrelation & dimensionality reduction)
# ======================================
pca = PCA(n_components=8, random_state=42)
X_pca = pca.fit_transform(X)
explained_var = np.sum(pca.explained_variance_ratio_)
print(f"🔹 PCA reduced dimensions from {X.shape[1]} → {X_pca.shape[1]}")
print(f"Explained Variance: {explained_var:.3f}")

# PCA Scree Plot
plt.figure(figsize=(7,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.show()

# ======================================
# 5. Elbow Method to find best k
# ======================================
inertias = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=1000)
    km.fit(X_pca)
    inertias.append(km.inertia_)

plt.figure(figsize=(7,4))
plt.plot(K, inertias, 'bo-')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# ======================================
# 6. Silhouette Method
# ======================================
sil_scores = []
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=1000)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    sil_scores.append(sil)

plt.figure(figsize=(7,4))
plt.plot(K, sil_scores, 'ro-')
plt.title("Silhouette Scores by k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

best_k = K[np.argmax(sil_scores)]
print(f"🏆 Best k by silhouette: {best_k}, score={max(sil_scores):.4f}")

# ======================================
# 7. Final KMeans with optimal k
# ======================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50, max_iter=1000)
labels = kmeans.fit_predict(X_pca)
data['Persona'] = labels
silhouette_avg = silhouette_score(X_pca, labels)
print(f"✅ Final silhouette score: {silhouette_avg:.3f}")

# ======================================
# 8. Silhouette Plot
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
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")
plt.title(f"Silhouette Plot (k={best_k}, score={silhouette_avg:.3f})")
plt.show()

# ======================================
# 9. 2D Visualization of Clusters
# ======================================
pca_2d = PCA(n_components=2, random_state=42)
X_vis = pca_2d.fit_transform(X_pca)
data['PCA1'] = X_vis[:, 0]
data['PCA2'] = X_vis[:, 1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Persona', palette='tab10', s=60)
plt.title(f"Shopper Persona Clusters (k={best_k}, silhouette={silhouette_avg:.3f})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Persona')
plt.show()

# ======================================
# 10. Save clustered data
# ======================================

