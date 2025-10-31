from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime
import joblib

app = Flask(__name__)
CORS(app)

# Initialize ML models
class ShopperPredictor:
    def __init__(self):
        self.purchase_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.persona_model = KMeans(n_clusters=5, random_state=42)
        self.is_trained = False
        
    def train_models(self, X, y):
        """Train purchase prediction and persona clustering models"""
        self.purchase_model.fit(X, y)
        self.persona_model.fit(X)
        self.is_trained = True
        
    def predict_purchase(self, features):
        """Predict if user will make a purchase"""
        if not self.is_trained:
            return self._fallback_prediction(features)
        
        prob = self.purchase_model.predict_proba([features])[0][1]
        will_buy = prob > 0.5
        confidence = int(prob * 100)
        
        return {
            'will_buy': bool(will_buy),
            'confidence': confidence,
            'probability': float(prob)
        }
    
    def _fallback_prediction(self, features):
        """Simple rule-based prediction when model isn't trained"""
        time_on_site, pages_visited, cart_value, prev_purchases, device = features
        
        score = 0
        score += min(time_on_site / 60, 10) * 0.15
        score += min(pages_visited, 20) * 0.1
        score += min(cart_value / 50, 10) * 0.2
        score += prev_purchases * 0.3
        score += 0.15 if device == 2 else 0.1  # desktop=2, mobile=1, tablet=0
        
        return {
            'will_buy': score > 0.5,
            'confidence': int(score * 100),
            'probability': float(score)
        }
    
    def identify_persona(self, features):
        """Identify user persona cluster"""
        personas = {
            0: {'type': 'Browser', 'color': 'gray'},
            1: {'type': 'Research Shopper', 'color': 'blue'},
            2: {'type': 'Quick Buyer', 'color': 'green'},
            3: {'type': 'VIP Loyalist', 'color': 'purple'},
            4: {'type': 'Casual Visitor', 'color': 'yellow'}
        }
        
        if self.is_trained:
            cluster = int(self.persona_model.predict([features])[0])
        else:
            cluster = self._fallback_persona(features)
        
        return personas.get(cluster, personas[4])
    
    def _fallback_persona(self, features):
        """Rule-based persona identification"""
        time_on_site, pages_visited, cart_value, prev_purchases, device = features
        
        if prev_purchases > 5 and cart_value > 100:
            return 3  # VIP Loyalist
        elif time_on_site > 300 and pages_visited > 10:
            return 1  # Research Shopper
        elif cart_value > 150 and time_on_site < 180:
            return 2  # Quick Buyer
        elif pages_visited < 3 and time_on_site < 60:
            return 0  # Browser
        else:
            return 4  # Casual Visitor

# Initialize predictor
predictor = ShopperPredictor()

# Generate synthetic training data
def generate_training_data(n_samples=1000):
    """Generate synthetic data for model training"""
    np.random.seed(42)
    
    data = {
        'time_on_site': np.random.randint(30, 600, n_samples),
        'pages_visited': np.random.randint(1, 20, n_samples),
        'cart_value': np.random.randint(20, 300, n_samples),
        'previous_purchases': np.random.randint(0, 10, n_samples),
        'device_type': np.random.randint(0, 3, n_samples)  # 0=tablet, 1=mobile, 2=desktop
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels based on features
    purchase_score = (
        (df['time_on_site'] / 600) * 0.2 +
        (df['pages_visited'] / 20) * 0.15 +
        (df['cart_value'] / 300) * 0.3 +
        (df['previous_purchases'] / 10) * 0.35
    )
    
    labels = (purchase_score > 0.5).astype(int)
    
    return df.values, labels

# Train models on startup
X_train, y_train = generate_training_data()
predictor.train_models(X_train, y_train)

def recommend_incentive(prediction, persona, session_data):
    """Recommend personalized incentive based on prediction and persona"""
    cart_value = session_data.get('cart_value', 0)
    prev_purchases = session_data.get('previous_purchases', 0)
    
    if not prediction['will_buy']:
        if persona['type'] == 'Research Shopper':
            return {
                'type': 'urgency',
                'message': '⏰ Only 2 left in stock! Complete purchase in 10 mins',
                'discount': 0,
                'priority': 'high'
            }
        elif cart_value > 100:
            return {
                'type': 'discount',
                'message': '💰 Get 15% OFF your cart - Limited time!',
                'discount': 15,
                'priority': 'high'
            }
        elif prev_purchases > 2:
            return {
                'type': 'loyalty',
                'message': '⭐ Earn 500 bonus points on this purchase!',
                'discount': 0,
                'priority': 'medium'
            }
        else:
            return {
                'type': 'discount',
                'message': '🎉 First-time special: 10% OFF your order',
                'discount': 10,
                'priority': 'medium'
            }
    else:
        if persona['type'] == 'VIP Loyalist':
            return {
                'type': 'loyalty',
                'message': '👑 VIP Bonus: Double points on this order!',
                'discount': 0,
                'priority': 'low'
            }
        else:
            return {
                'type': 'none',
                'message': '✅ Great choice! Proceeding to checkout...',
                'discount': 0,
                'priority': 'low'
            }

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        # Extract features
        device_map = {'tablet': 0, 'mobile': 1, 'desktop': 2}
        features = [
            data.get('time_on_site', 0),
            data.get('pages_visited', 0),
            data.get('cart_value', 0),
            data.get('previous_purchases', 0),
            device_map.get(data.get('device_type', 'mobile'), 1)
        ]
        
        # Make predictions
        prediction = predictor.predict_purchase(features)
        persona = predictor.identify_persona(features)
        incentive = recommend_incentive(prediction, persona, data)
        
        response = {
            'session_id': data.get('session_id', datetime.now().timestamp()),
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'persona': persona,
            'incentive': incentive,
            'features': {
                'time_on_site': features[0],
                'pages_visited': features[1],
                'cart_value': features[2],
                'previous_purchases': features[3],
                'device_type': data.get('device_type', 'mobile')
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Get analytics summary"""
    # In production, this would query a database
    return jsonify({
        'total_sessions': 150,
        'conversion_rate': 67,
        'avg_session_time': 245,
        'top_persona': 'Research Shopper',
        'total_revenue': 45670,
        'incentives_used': {
            'discount': 45,
            'loyalty': 32,
            'urgency': 28
        }
    }), 200

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain models with new data"""
    try:
        data = request.get_json()
        X = np.array(data.get('features', []))
        y = np.array(data.get('labels', []))
        
        if len(X) > 0 and len(y) > 0:
            predictor.train_models(X, y)
            return jsonify({'message': 'Models retrained successfully'}), 200
        else:
            return jsonify({'error': 'Invalid training data'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': predictor.is_trained,
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)