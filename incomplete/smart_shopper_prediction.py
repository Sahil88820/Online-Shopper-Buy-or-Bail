"""
Smart Shopper AI - Purchase Prediction using CatBoost/XGBoost
Binary classification to predict Buy vs Bail with SHAP explainability
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class PurchasePredictor:
    """Predict purchase probability using gradient boosting"""
    
    def __init__(self, model_type='catboost', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        
    def initialize_model(self):
        """Initialize CatBoost or XGBoost model"""
        if self.model_type == 'catboost':
            self.model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=self.random_state,
                verbose=False,
                eval_metric='AUC',
                early_stopping_rounds=50
            )
        else:  # xgboost
            self.model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                reg_lambda=3,
                random_state=self.random_state,
                eval_metric='auc',
                early_stopping_rounds=50
            )
        
        print(f"✅ Initialized {self.model_type.upper()} model")
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the purchase prediction model"""
        print("\n" + "="*60)
        print(f"TRAINING {self.model_type.upper()} PURCHASE PREDICTION MODEL")
        print("="*60)
        
        if self.model is None:
            self.initialize_model()
        
        # Train model
        if X_val is not None and y_val is not None:
            if self.model_type == 'catboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    verbose=False
                )
            else:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
        else:
            self.model.fit(X_train, y_train)
        
        print("\n✅ Training completed!")
        
        return self.model
    
    def evaluate(self, X_test, y_test, dataset_name='Test'):
        """Evaluate model performance"""
        print(f"\n📊 Evaluating on {dataset_name} Set")
        print("-" * 40)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Bail', 'Buy']))
        
        return metrics, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bail', 'Buy'],
            yticklabels=['Bail', 'Buy'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n📊 Confusion matrix saved to outputs/confusion_matrix.png")
    
    def plot_feature_importance(self, feature_names, top_n=20):
        """Plot feature importance"""
        if self.model_type == 'catboost':
            importance = self.model.get_feature_importance()
        else:
            importance = self.model.feature_importances_
        
        # Create DataFrame
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_imp
        
        # Plot top N features
        plt.figure(figsize=(10, 8))
        top_features = feature_imp.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        print("📊 Feature importance plot saved to outputs/feature_importance.png")
        
        return feature_imp
    
    def explain_with_shap(self, X_sample, feature_names, max_display=20):
        """Generate SHAP values for model explainability"""
        print("\n🔍 Generating SHAP explanations...")
        
        # Create SHAP explainer
        if self.model_type == 'catboost':
            explainer = shap.TreeExplainer(self.model)
        else:
            explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig('outputs/shap_summary.png', dpi=300, bbox_inches='tight')
        print("📊 SHAP summary plot saved to outputs/shap_summary.png")
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig('outputs/shap_bar.png', dpi=300, bbox_inches='tight')
        print("📊 SHAP bar plot saved to outputs/shap_bar.png")
        plt.close()
        
        return shap_values, explainer
    
    def predict_single(self, features, feature_names):
        """Make prediction for a single shopper"""
        if isinstance(features, dict):
            X = pd.DataFrame([features])[feature_names]
        else:
            X = pd.DataFrame([features], columns=feature_names)
        
        pred_proba = self.model.predict_proba(X)[0]
        pred_class = self.model.predict(X)[0]
        
        return {
            'will_buy': bool(pred_class),
            'buy_probability': float(pred_proba[1]),
            'bail_probability': float(pred_proba[0]),
            'confidence': float(max(pred_proba)) * 100
        }
    
    def save_model(self, output_dir='models'):
        """Save trained model"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = f'{output_dir}/purchase_predictor_{self.model_type}.pkl'
        joblib.dump(self.model, model_path)
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                f'{output_dir}/feature_importance.csv',
                index=False
            )
        
        print(f"\n💾 Model saved to {model_path}")


# Example usage
if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # Load preprocessed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    feature_names = joblib.load('data/processed/feature_names.pkl')
    
    print("="*60)
    print("SMART SHOPPER AI - PURCHASE PREDICTION")
    print("="*60)
    
    # Initialize predictor (choose 'catboost' or 'xgboost')
    predictor = PurchasePredictor(model_type='catboost', random_state=42)
    
    # Train model
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = predictor.evaluate(X_test, y_test)
    
    # Plot confusion matrix
    predictor.plot_confusion_matrix(y_test, y_pred)
    
    # Plot feature importance
    feature_imp = predictor.plot_feature_importance(feature_names, top_n=20)
    
    # SHAP explanations (on sample of test data)
    sample_size = min(100, len(X_test))
    shap_values, explainer = predictor.explain_with_shap(
        X_test.sample(sample_size, random_state=42),
        feature_names,
        max_display=20
    )
    
    # Save model
    predictor.save_model('models')
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    sample_shopper = X_test.iloc[0]
    result = predictor.predict_single(sample_shopper, feature_names)
    print(f"Will Buy: {result['will_buy']}")
    print(f"Buy Probability: {result['buy_probability']:.2%}")
    print(f"Confidence: {result['confidence']:.1f}%")
    
    print("\n✅ Purchase prediction model training completed!")
