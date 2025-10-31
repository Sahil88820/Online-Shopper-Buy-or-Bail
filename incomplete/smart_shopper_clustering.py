"""
Smart Shopper AI - Unsupervised Learning for Persona Clustering
Uses KMeans clustering and PCA for dimensionality reduction
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class PersonaClusterer:
    """Identify shopper behavioral personas using KMeans clustering"""
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.pca = None
        self.persona_profiles = {}
        
    def find_optimal_clusters(self, X, max_clusters=10):
        """Use elbow method and silhouette score to find optimal k"""
        print("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/cluster_optimization.png', dpi=300, bbox_inches='tight')
        print("\n📊 Cluster optimization plot saved to outputs/cluster_optimization.png")
        
        return silhouette_scores
    
    def fit_clusters(self, X):
        """Fit KMeans clustering model"""
        print(f"\nFitting KMeans with {self.n_clusters} clusters...")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=20,
            max_iter=300
        )
        
        clusters = self.kmeans.fit_predict(X)
        
        # Calculate metrics
        silhouette = silhouette_score(X, clusters)
        davies_bouldin = davies_bouldin_score(X, clusters)
        
        print(f"✅ Clustering complete!")
        print(f"   Silhouette Score: {silhouette:.3f}")
        print(f"   Davies-Bouldin Index: {davies_bouldin:.3f}")
        print(f"   Cluster distribution: {np.bincount(clusters)}")
        
        return clusters
    
    def reduce_dimensions(self, X, n_components=2):
        """Apply PCA for visualization"""
        print(f"\nApplying PCA (n_components={n_components})...")
        
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X)
        
        explained_var = self.pca.explained_variance_ratio_
        print(f"   Explained variance: {explained_var}")
        print(f"   Total variance explained: {sum(explained_var):.2%}")
        
        return X_pca
    
    def profile_personas(self, X, clusters, feature_names):
        """Create detailed profiles for each persona"""
        print("\nProfiling personas...")
        
        df = pd.DataFrame(X, columns=feature_names)
        df['Cluster'] = clusters
        
        persona_names = {
            0: 'Deal Hunter',
            1: 'Impulse Buyer',
            2: 'Window Browser',
            3: 'Research Shopper',
            4: 'Loyal Customer'
        }
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['Cluster'] == cluster_id].drop('Cluster', axis=1)
            
            # Calculate statistics
            profile = {
                'name': persona_names.get(cluster_id, f'Cluster {cluster_id}'),
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,
                'mean_values': cluster_data.mean().to_dict(),
                'characteristics': []
            }
            
            # Identify key characteristics (features with high/low values)
            mean_all = df.drop('Cluster', axis=1).mean()
            mean_cluster = cluster_data.mean()
            
            # Find distinguishing features
            diff = ((mean_cluster - mean_all) / mean_all * 100).abs()
            top_features = diff.nlargest(5)
            
            for feature, pct_diff in top_features.items():
                value = mean_cluster[feature]
                overall = mean_all[feature]
                direction = "higher" if value > overall else "lower"
                profile['characteristics'].append(
                    f"{feature}: {value:.2f} ({pct_diff:.1f}% {direction} than average)"
                )
            
            self.persona_profiles[cluster_id] = profile
            
            print(f"\n🎭 {profile['name']} (Cluster {cluster_id})")
            print(f"   Size: {profile['size']} ({profile['percentage']:.1f}%)")
            print(f"   Key characteristics:")
            for char in profile['characteristics']:
                print(f"      • {char}")
        
        return self.persona_profiles
    
    def visualize_clusters(self, X_pca, clusters):
        """Create 2D visualization of clusters"""
        print("\nCreating cluster visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Color palette
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # Plot each cluster
        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            persona_name = self.persona_profiles[cluster_id]['name']
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=colors[cluster_id],
                label=f'{persona_name} ({sum(mask)})',
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
        
        # Plot cluster centers
        centers_pca = self.pca.transform(self.kmeans.cluster_centers_)
        plt.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c='black',
            marker='X',
            s=300,
            edgecolors='white',
            linewidth=2,
            label='Centroids'
        )
        
        plt.xlabel('First Principal Component', fontsize=12)
        plt.ylabel('Second Principal Component', fontsize=12)
        plt.title('Shopper Behavioral Personas (KMeans + PCA)', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('outputs/persona_clusters.png', dpi=300, bbox_inches='tight')
        print("📊 Cluster visualization saved to outputs/persona_clusters.png")
        
    def assign_persona_names(self):
        """Assign meaningful names based on cluster characteristics"""
        # This is a simplified version - in practice, analyze cluster profiles
        persona_mapping = {
            0: {
                'name': 'Deal Hunter',
                'description': 'Price-sensitive shoppers who browse extensively for deals',
                'traits': ['High page views', 'Low page values', 'High bounce rate']
            },
            1: {
                'name': 'Impulse Buyer',
                'description': 'Quick decision makers with high conversion rates',
                'traits': ['Short sessions', 'High page values', 'Low bounce rate']
            },
            2: {
                'name': 'Window Browser',
                'description': 'Casual visitors with low purchase intent',
                'traits': ['Medium page views', 'Low engagement', 'High exit rate']
            },
            3: {
                'name': 'Research Shopper',
                'description': 'Thorough researchers who spend time comparing options',
                'traits': ['Long sessions', 'Many pages', 'Product-focused']
            },
            4: {
                'name': 'Loyal Customer',
                'description': 'Returning visitors with high purchase intent',
                'traits': ['Returning visitor', 'High page values', 'Low bounce']
            }
        }
        return persona_mapping
    
    def save_model(self, output_dir='models'):
        """Save clustering model and PCA transformer"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.kmeans, f'{output_dir}/kmeans_model.pkl')
        joblib.dump(self.pca, f'{output_dir}/pca_transformer.pkl')
        joblib.dump(self.persona_profiles, f'{output_dir}/persona_profiles.pkl')
        
        print(f"\n💾 Models saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Load preprocessed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    feature_names = joblib.load('data/processed/feature_names.pkl')
    
    print("="*60)
    print("SMART SHOPPER AI - PERSONA CLUSTERING")
    print("="*60)
    
    # Initialize clusterer
    clusterer = PersonaClusterer(n_clusters=5, random_state=42)
    
    # Find optimal clusters (optional)
    # clusterer.find_optimal_clusters(X_train, max_clusters=10)
    
    # Fit clustering model
    clusters = clusterer.fit_clusters(X_train)
    
    # Apply PCA for visualization
    X_pca = clusterer.reduce_dimensions(X_train, n_components=2)
    
    # Profile personas
    personas = clusterer.profile_personas(X_train, clusters, feature_names)
    
    # Visualize clusters
    clusterer.visualize_clusters(X_pca, clusters)
    
    # Save model
    clusterer.save_model('models')
    
    print("\n✅ Persona clustering completed successfully!")
