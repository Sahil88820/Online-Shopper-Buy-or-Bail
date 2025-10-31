# 🎯 AI Shopper Behavior Predictor

A comprehensive web application that predicts online shopper behavior using machine learning, identifies persona clusters, and recommends personalized incentives in real-time.

## 🚀 Features

- **ML-Powered Predictions**: Random Forest classifier predicts purchase likelihood with confidence scores
- **Persona Clustering**: K-Means clustering identifies 5 distinct shopper personas
- **Personalized Incentives**: Smart recommendation engine for discounts, loyalty points, and urgency triggers
- **Live Dashboard**: Real-time analytics with conversion rates, session metrics, and persona distribution
- **Multi-Language Architecture**: Separate implementations for Python (backend), JavaScript (frontend), and SQL (database)

## 📁 File Structure

```
shopper-predictor/
├── backend/
│   └── app.py                 # Python Flask API with ML models
├── frontend/
│   ├── index.html             # Main HTML structure
│   └── shopper-predictor.js   # JavaScript frontend logic
├── database/
│   └── schema.sql             # PostgreSQL database schema
└── README.md                  # This file
```

## 🛠️ Tech Stack

### Backend (Python)
- **Flask**: Web framework for API endpoints
- **scikit-learn**: ML models (RandomForest, KMeans)
- **pandas/numpy**: Data processing
- **Flask-CORS**: Cross-origin resource sharing

### Frontend (JavaScript)
- **Vanilla JavaScript**: Core functionality
- **Fetch API**: Backend communication
- **CSS3**: Modern styling with gradients and animations

### Database (SQL)
- **PostgreSQL**: Primary database
- Comprehensive schema with 10+ tables
- Optimized indexes and views
- Stored procedures for analytics

## 📦 Installation

### Prerequisites
- Python 3.8+
- Node.js (optional, for development)
- PostgreSQL 14+

### Backend Setup

1. **Create virtual environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install flask flask-cors scikit-learn pandas numpy joblib
```

3. **Run the Flask server**
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Open the HTML file**
```bash
cd frontend
# Simply open index.html in a browser, or use a local server:
python -m http.server 8000
```

2. **Access the app**
Navigate to `http://localhost:8000` in your browser

### Database Setup

1. **Create database**
```bash
psql -U postgres
CREATE DATABASE shopper_predictor;
\c shopper_predictor
```

2. **Run schema**
```bash
psql -U postgres -d shopper_predictor -f database/schema.sql
```

## 🔌 API Endpoints

### POST /api/predict
Predict shopper behavior and recommend incentives

**Request:**
```json
{
  "session_id": 1234567890,
  "time_on_site": 245,
  "pages_visited": 8,
  "cart_value": 150.50,
  "previous_purchases": 3,
  "device_type": "desktop"
}
```

**Response:**
```json
{
  "session_id": 1234567890,
  "timestamp": "2025-10-31T10:30:00",
  "prediction": {
    "will_buy": true,
    "confidence": 78,
    "probability": 0.78
  },
  "persona": {
    "type": "Research Shopper",
    "color": "blue"
  },
  "incentive": {
    "type": "loyalty",
    "message": "⭐ Earn 500 bonus points on this purchase!",
    "discount": 0,
    "priority": "medium"
  }
}
```

### GET /api/analytics
Get real-time analytics summary

**Response:**
```json
{
  "total_sessions": 150,
  "conversion_rate": 67,
  "avg_session_time": 245,
  "top_persona": "Research Shopper",
  "total_revenue": 45670
}
```

### GET /api/health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_trained": true,
  "timestamp": "2025-10-31T10:30:00"
}
```

## 🧠 ML Models

### Purchase Prediction (Random Forest)
- **Features**: time_on_site, pages_visited, cart_value, previous_purchases, device_type
- **Output**: Binary classification (will buy / will bail) with confidence score
- **Training**: 1000 synthetic samples with realistic patterns

### Persona Clustering (K-Means)
- **Clusters**: 5 distinct personas
  - **VIP Loyalist**: High-value repeat customers
  - **Research Shopper**: Thorough browsers with high engagement
  - **Quick Buyer**: Fast decision makers with high cart values
  - **Browser**: Low engagement, window shopping
  - **Casual Visitor**: Average engagement, exploring

### Incentive Recommendation Engine
Rules-based system that considers:
- Purchase prediction confidence
- Persona type
- Cart value
- Purchase history
- Session behavior

## 📊 Database Schema

### Core Tables
- **users**: Customer profiles and loyalty tiers
- **sessions**: Browsing session tracking
- **carts**: Shopping cart management
- **predictions**: ML prediction history
- **personas**: Persona assignments
- **incentives**: Incentive recommendations and conversions
- **conversions**: Successful purchases
- **analytics_snapshots**: Periodic analytics

### Views
- **dashboard_realtime**: 24-hour rolling analytics
- **incentive_performance**: Conversion rates by incentive type
- **persona_distribution**: Persona analytics and lifetime value

## 🎨 Customization

### Adding New Personas
Modify the persona identification logic in both:
- `app.py`: `ShopperPredictor._fallback_persona()`
- `shopper-predictor.js`: `APIClient.identifyPersona()`

### Adjusting Prediction Weights
Update feature weights in:
- `app.py`: `ShopperPredictor._fallback_prediction()`
- `shopper-predictor.js`: `APIClient.fallbackPredict()`

### Creating New Incentive Types
Add new incentive logic in:
- `app.py`: `recommend_incentive()` function
- `shopper-predictor.js`: `APIClient.recommendIncentive()`

## 🧪 Testing

### Backend Tests
```bash
# Test prediction endpoint
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_on_site": 300,
    "pages_visited": 10,
    "cart_value": 120,
    "previous_purchases": 5,
    "device_type": "desktop"
  }'

# Test health endpoint
curl http://localhost:5000/api/health
```

### Frontend Tests
Open browser console and check for:
- WebSocket connections
- API calls every 5 seconds
- Session rendering
- Analytics updates

## 🚢 Deployment

### Backend (Heroku)
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create shopper-predictor-api
git push heroku main
```

### Frontend (Vercel/Netlify)
```bash
# Update API_BASE_URL in shopper-predictor.js
const API_BASE_URL = 'https://your-api.herokuapp.com/api';

# Deploy
vercel deploy
# or
netlify deploy
```

### Database (Heroku Postgres)
```bash
heroku addons:create heroku-postgresql:hobby-dev
heroku pg:psql < database/schema.sql
```

## 📈 Performance Optimization

- **Caching**: Implement Redis for session data
- **Database Indexing**: Already optimized with indexes on key columns
- **Model Optimization**: Use joblib to save/load trained models
- **Frontend**: Implement virtual scrolling for large session lists

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License - feel free to use this project for commercial or personal use

## 🆘 Support

For issues or questions:
- Open an issue on GitHub
- Email: support@shopper-predictor.com
- Documentation: https://docs.shopper-predictor.com

## 🎯 Roadmap

- [ ] Deep learning models (LSTM for sequence prediction)
- [ ] A/B testing framework for incentives
- [ ] Real-time WebSocket updates
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Integration with e-commerce platforms (Shopify, WooCommerce)

---

Built with ❤️ for better e-commerce conversions