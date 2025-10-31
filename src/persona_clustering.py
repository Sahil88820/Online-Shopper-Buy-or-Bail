"""
KMeans clustering module for identifying shopper personas.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaClusterer:
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.pca = PCA(n_components=2)
        self.persona_profiles = {}
        
    def fit(self, X):
        """Fit KMeans clustering model."""
        try:
            self.kmeans.fit(X)
            logger.info("Successfully fitted KMeans model")
            self._create_persona_profiles(X)
            return self
        except Exception as e:
            logger.error(f"Error fitting KMeans model: {str(e)}")
            raise
            
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self.kmeans.predict(X)
        
    def transform_pca(self, X):
        """Transform data to 2D for visualization."""
        return self.pca.fit_transform(X)
        
    def _create_persona_profiles(self, X):
        """Create profiles for each persona based on cluster centers."""
        centers = self.kmeans.cluster_centers_
        for i in range(self.n_clusters):
            self.persona_profiles[i] = {
                'center': centers[i],
                'size': sum(self.kmeans.labels_ == i),
                'description': self._generate_persona_description(centers[i], X.columns)
            }
            
    def _generate_persona_description(self, center, feature_names):
        """Generate descriptive text for each persona."""
        # This is a simplified version - you would want to make this more sophisticated
        description = "Shopper characterized by:\n"
        for feature, value in zip(feature_names, center):
            if abs(value) > 0.5:  # Only include significant features
                direction = "high" if value > 0 else "low"
                description += f"- {direction} {feature}\n"
        return description
        
    def save_model(self, output_dir):
        """Save clustering model and related objects."""
        with open(f"{output_dir}/kmeans_model.pkl", 'wb') as f:
            pickle.dump(self.kmeans, f)
        with open(f"{output_dir}/pca_transformer.pkl", 'wb') as f:
            pickle.dump(self.pca, f)
        with open(f"{output_dir}/persona_profiles.pkl", 'wb') as f:
            pickle.dump(self.persona_profiles, f)
            
    def load_model(self, input_dir):
        """Load clustering model and related objects."""
        with open(f"{input_dir}/kmeans_model.pkl", 'rb') as f:
            self.kmeans = pickle.load(f)
        with open(f"{input_dir}/pca_transformer.pkl", 'rb') as f:
            self.pca = pickle.load(f)
        with open(f"{input_dir}/persona_profiles.pkl", 'rb') as f:
            self.persona_profiles = pickle.load(f)

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load preprocessed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    
    # Create and fit clusterer
    clusterer = PersonaClusterer(n_clusters=5)
    clusterer.fit(X_train)
    
    # Save model
    clusterer.save_model("models")