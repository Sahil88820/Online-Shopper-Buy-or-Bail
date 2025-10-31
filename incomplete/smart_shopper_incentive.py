"""
Smart Shopper AI - Incentive Recommendation Engine
Multi-class XGBoost classifier to recommend personalized incentives
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class IncentiveRecommender:
    """Recommend personalized incentives to prevent cart abandonment"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.label_encoder = LabelEncoder()
        self.incentive_types = [
            'discount_10',
            'discount_15',
            'discount_20',
            'loyalty_points',
            'free_shipping',
            'urgency_banner',
            'none'
        ]
        
    def generate_incentive_labels(self, X, y_pred_proba, clusters):
        """
        Generate synthetic incentive labels based on:
        - Purchase probability
        - Shopper persona
        - Session characteristics
        """
        print("Generating incentive recommendations...")
        
        incentives = []
        
        for idx, (prob, cluster) in enumerate(zip(y_pred_proba, clusters)):
            row = X.iloc[idx]
            
            # If high purchase probability, no incentive needed
            if prob > 0.7:
                incentive = 'none'
            
            # If low probability, need strong incentive
            elif prob < 0.3:
                # Deal Hunter or Window Browser - price sensitive
                if cluster in [0, 2]:
                    if row['PageValues'] > 50:
                        incentive = 'discount_20'
                    else:
                        incentive = 'discount_15'
                
                # Impulse Buyer - urgency works
                elif cluster == 1:
                    incentive = 'urgency_banner'
                
                # Research Shopper - needs time and information
                elif cluster == 3:
                    incentive = 'free_shipping'
                
                # Loyal Customer - reward loyalty
                elif cluster == 4:
                    incentive = 'loyalty_points'
                
                else:
                    incentive = 'discount_10'
            
            # Medium probability - moderate incentive
            else:
                if cluster in [0, 2]:  # Price sensitive
                    incentive = 'discount_10'
                elif cluster == 4:  # Loyal
                    incentive = 'loyalty_points'
                elif row['BounceRates'] > 0.5:  # High bounce
                    incentive = 'urgency_banner'
                else:
                    incentive = 'free_shipping'
            
            incentives.append(incentive)
        
        return np.array(incentives)
    
    def prepare_training_data(self, X, y_pred_proba, clusters, personas):
        """Prepare data for incentive model training"""
        print("\nPreparing incentive training data...")
        
        # Generate incentive labels
        incentive_labels = self.generate_incentive_labels(X, y_pred_proba, clusters)
        
        # Add cluster and probability as features
        X_incentive = X.copy()
        X_incentive['cluster'] = clusters
        X_incentive['buy_probability'] = y_pred_proba
        X_incentive['bail_probability'] = 1 - y_pred_proba
        
        # Encode labels
        y_incentive = self.label_encoder.fit_transform(incentive_labels)
        
        print(f"Incentive distribution:")
        for incentive, count in zip(*np.unique(incentive_labels, return_counts=True)):
            print(f"  {incentive}: {count} ({count/len(incentive_labels)*100:.1f}%)")
        
        return X_incentive, y_incentive, incentive_labels
    
    def train(self, X_train, y_train):
        """Train XGBoost multi-class classifier for incentive recommendation"""
        print("\n" + "="*60)
        print("TRAINING INCENTIVE RECOMMENDATION MODEL")
        print("="*60)
        
        self.model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            objective='multi:softmax',
            num_class=len(self.label_encoder.classes_),
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        self.model.fit(X_train, y_train)
        
        print("\n✅ Training completed!")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate incentive recommendation model"""
        print("\n📊 Evaluating Incentive Model")
        print("-" * 40)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        
        # Convert numeric labels back to text
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        print("\nClassification Report:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        return accuracy, y_pred
    
    def plot_incentive_distribution(self, y_true, y_pred):
        """Plot distribution of recommended incentives"""
        true_labels = self.label_encoder.inverse_transform(y_true)
        pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # True distribution
        true_counts = pd.Series(true_labels).value_counts()
        ax1.bar(range(len(true_counts)), true_counts.values, color='steelblue')
        ax1.set_xticks(range(len(true_counts)))
        ax1.set_xticklabels(true_counts.index, rotation=45, ha='right')
        ax1.set_title('True Incentive Distribution', fontweight='bold')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        
        # Predicted distribution
        pred_counts = pd.Series(pred_labels).value_counts()
        ax2.bar(range(len(pred_counts)), pred_counts.values, color='coral')
        ax2.set_xticks(range(len(pred_counts)))
        ax2.set_xticklabels(pred_counts.index, rotation=45, ha='right')
        ax2.set_title('Predicted Incentive Distribution', fontweight='bold')
        ax2.set_ylabel('Count')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/incentive_distribution.png', dpi=300, bbox_inches='tight')
        print("\n📊 Incentive distribution plot saved to outputs/incentive_distribution.png")
    
    def recommend_incentive(self, features, buy_probability, cluster):
        """Recommend incentive for a single shopper"""
        # Prepare features
        if isinstance(features, pd.Series):
            X = features.to_frame().T
        else:
            X = pd.DataFrame([features])
        
        X['cluster'] = cluster
        X['buy_probability'] = buy_probability
        X['bail_probability'] = 1 - buy_probability
        
        # Predict incentive
        incentive_encoded = self.model.predict(X)[0]
        incentive_type = self.label_encoder.inverse_transform([incentive_encoded])[0]
        
        # Get probabilities for all incentives
        incentive_probs = self.model.predict_proba(X)[0]
        
        # Map incentive to message and details
        incentive_details = self._get_incentive_details(incentive_type)
        
        return {
            'incentive_type': incentive_type,
            'confidence': float(max(incentive_probs)) * 100,
            'all_probabilities': {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(incentive_probs)
            },
            **incentive_details
        }
    
    def _get_incentive_details(self, incentive_type):
        """Get detailed message and parameters for each incentive type"""
        incentive_map = {
            'discount_10': {
                'message': '🎉 Get 10% OFF your order - Limited time!',
                'discount_percent': 10,
                'category': 'discount'
            },
            'discount_15': {
                'message': '💰 Save 15% on your cart - Don\'t miss out!',
                'discount_percent': 15,
                'category': 'discount'
            },
            'discount_20': {
                'message': '🔥 SPECIAL: 20% OFF - Expires soon!',
                'discount_percent': 20,
                'category': 'discount'
            },
            'loyalty_points': {
                'message': '⭐ Earn 500 bonus loyalty points on this purchase!',
                'points': 500,
                'category': 'loyalty'
            },
            'free_shipping': {
                'message': '🚚 FREE Shipping on your order - Today only!',
                'shipping_cost': 0,
                'category': 'shipping'
            },
            'urgency_banner': {
                'message': '⏰ Only 2 left in stock! Order in next 10 minutes',
                'urgency_level': 'high',
                'category': 'urgency'
            },
            'none': {
                'message': '✅ Great choice! Proceeding to checkout...',
                'category': 'none'
            }
        }
        
        return incentive_map.get(incentive_type, incentive_map['none'])
    
    def save_model(self, output_dir='models'):
        """Save incentive recommendation model"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.model, f'{output_dir}/incentive_recommender.pkl')
        joblib.dump(self.label_encoder, f'{output_dir}/incentive_label_encoder.pkl')
        
        print(f"\n💾 Incentive model saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Load preprocessed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    feature_names = joblib.load('data/processed/feature_names.pkl')
    
    # Load purchase prediction model to get probabilities
    purchase_model = joblib.load('models/purchase_predictor_catboost.pkl')
    train_proba = purchase_model.predict_proba(X_train)[:, 1]
    test_proba = purchase_model.predict_proba(X_test)[:, 1]
    
    # Load cluster assignments
    kmeans = joblib.load('models/kmeans_model.pkl')
    train_clusters = kmeans.predict(X_train)
    test_clusters = kmeans.predict(X_test)
    
    print("="*60)
    print("SMART SHOPPER AI - INCENTIVE RECOMMENDATION")
    print("="*60)
    
    # Initialize recommender
    recommender = IncentiveRecommender(random_state=42)
    
    # Prepare training data
    X_train_inc, y_train_inc, train_labels = recommender.prepare_training_data(
        X_train, train_proba, train_clusters, None
    )
    
    X_test_inc, y_test_inc, test_labels = recommender.prepare_training_data(
        X_test, test_proba, test_clusters, None
    )
    
    # Train model
    recommender.train(X_train_inc, y_train_inc)
    
    # Evaluate model
    accuracy, y_pred = recommender.evaluate(X_test_inc, y_test_inc)
    
    # Plot distributions
    recommender.plot_incentive_distribution(y_test_inc, y_pred)
    
    # Save model
    recommender.save_model('models')
    
    # Example recommendation
    print("\n" + "="*60)
    print("EXAMPLE INCENTIVE RECOMMENDATION")
    print("="*60)
    sample_features = X_test.iloc[0]
    sample_prob = test_proba[0]
    sample_cluster = test_clusters[0]
    
    recommendation = recommender.recommend_incentive(
        sample_features, sample_prob, sample_cluster
    )
    
    print(f"Incentive Type: {recommendation['incentive_type']}")
    print(f"Message: {recommendation['message']}")
    print(f"Confidence: {recommendation['confidence']:.1f}%")
    
    print("\n✅ Incentive recommendation model training completed!")
