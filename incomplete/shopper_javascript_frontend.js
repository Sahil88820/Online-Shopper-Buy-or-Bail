// Shopper Behavior Predictor - Frontend JavaScript
// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// State Management
const state = {
  sessions: [],
  analytics: {
    totalSessions: 0,
    conversionRate: 0,
    avgSessionTime: 0,
    topPersona: 'N/A'
  },
  autoGenerate: true
};

// API Client
class APIClient {
  static async predict(sessionData) {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(sessionData)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Prediction error:', error);
      // Fallback to client-side prediction
      return this.fallbackPredict(sessionData);
    }
  }

  static fallbackPredict(data) {
    // Client-side fallback prediction
    const { time_on_site, pages_visited, cart_value, previous_purchases } = data;
    
    let score = 0;
    score += Math.min(time_on_site / 60, 10) * 0.15;
    score += Math.min(pages_visited, 20) * 0.1;
    score += Math.min(cart_value / 50, 10) * 0.2;
    score += previous_purchases * 0.3;
    
    const willBuy = score > 0.5;
    const confidence = Math.round(score * 100);
    
    return {
      prediction: { will_buy: willBuy, confidence },
      persona: this.identifyPersona(data),
      incentive: this.recommendIncentive(willBuy, data)
    };
  }

  static identifyPersona(data) {
    const { time_on_site, pages_visited, cart_value, previous_purchases } = data;
    
    if (previous_purchases > 5 && cart_value > 100) {
      return { type: 'VIP Loyalist', color: 'purple' };
    } else if (time_on_site > 300 && pages_visited > 10) {
      return { type: 'Research Shopper', color: 'blue' };
    } else if (cart_value > 150 && time_on_site < 180) {
      return { type: 'Quick Buyer', color: 'green' };
    } else if (pages_visited < 3 && time_on_site < 60) {
      return { type: 'Browser', color: 'gray' };
    } else {
      return { type: 'Casual Visitor', color: 'yellow' };
    }
  }

  static recommendIncentive(willBuy, data) {
    const { cart_value, previous_purchases } = data;
    
    if (!willBuy) {
      if (cart_value > 100) {
        return {
          type: 'discount',
          message: '💰 Get 15% OFF your cart - Limited time!',
          discount: 15
        };
      } else if (previous_purchases > 2) {
        return {
          type: 'loyalty',
          message: '⭐ Earn 500 bonus points on this purchase!',
          discount: 0
        };
      } else {
        return {
          type: 'discount',
          message: '🎉 First-time special: 10% OFF your order',
          discount: 10
        };
      }
    }
    return {
      type: 'none',
      message: '✅ Great choice! Proceeding to checkout...',
      discount: 0
    };
  }

  static async getAnalytics() {
    try {
      const response = await fetch(`${API_BASE_URL}/analytics`);
      return await response.json();
    } catch (error) {
      console.error('Analytics error:', error);
      return state.analytics;
    }
  }
}

// Session Generator
class SessionGenerator {
  static generate() {
    const deviceTypes = ['desktop', 'mobile', 'tablet'];
    
    return {
      session_id: Date.now(),
      time_on_site: Math.floor(Math.random() * 600) + 30,
      pages_visited: Math.floor(Math.random() * 20) + 1,
      cart_value: Math.floor(Math.random() * 300) + 20,
      previous_purchases: Math.floor(Math.random() * 10),
      device_type: deviceTypes[Math.floor(Math.random() * deviceTypes.length)],
      timestamp: new Date().toLocaleTimeString()
    };
  }
}

// UI Renderer
class UIRenderer {
  static renderAnalytics() {
    const { totalSessions, conversionRate, avgSessionTime, topPersona } = state.analytics;
    
    document.getElementById('total-sessions').textContent = totalSessions;
    document.getElementById('conversion-rate').textContent = `${conversionRate}%`;
    document.getElementById('avg-session-time').textContent = `${avgSessionTime}s`;
    document.getElementById('top-persona').textContent = topPersona;
  }

  static renderSession(session) {
    const sessionList = document.getElementById('session-list');
    const sessionCard = document.createElement('div');
    sessionCard.className = 'session-card';
    sessionCard.innerHTML = `
      <div class="session-header">
        <div class="session-info">
          <span class="session-time">${session.timestamp}</span>
          <span class="session-id">Session #${session.session_id.toString().slice(-6)}</span>
        </div>
        <div class="session-stats">
          <span>${session.time_on_site}s</span>
          <span>${session.pages_visited} pages</span>
          <span>$${session.cart_value}</span>
        </div>
      </div>
      
      <div class="session-body">
        <div class="prediction-section">
          <h4>ML Prediction</h4>
          <div class="prediction ${session.prediction.will_buy ? 'positive' : 'negative'}">
            <span class="icon">${session.prediction.will_buy ? '✓' : '✗'}</span>
            <span>${session.prediction.will_buy ? 'Will Buy' : 'Will Bail'}</span>
            <span class="confidence">${session.prediction.confidence}%</span>
          </div>
        </div>
        
        <div class="persona-section">
          <h4>Persona Cluster</h4>
          <div class="persona persona-${session.persona.color}">
            ${session.persona.type}
          </div>
        </div>
        
        <div class="incentive-section">
          <h4>Recommended Incentive</h4>
          <div class="incentive incentive-${session.incentive.type}">
            <p>${session.incentive.message}</p>
            ${session.incentive.discount > 0 ? `<span class="discount">-${session.incentive.discount}%</span>` : ''}
          </div>
        </div>
      </div>
    `;
    
    sessionList.insertBefore(sessionCard, sessionList.firstChild);
    
    // Keep only last 10 sessions
    while (sessionList.children.length > 10) {
      sessionList.removeChild(sessionList.lastChild);
    }
  }

  static showLoading() {
    document.getElementById('loading-indicator').style.display = 'block';
  }

  static hideLoading() {
    document.getElementById('loading-indicator').style.display = 'none';
  }
}

// Analytics Calculator
class AnalyticsCalculator {
  static calculate() {
    if (state.sessions.length === 0) return;

    const conversions = state.sessions.filter(s => s.prediction.will_buy).length;
    const avgTime = state.sessions.reduce((acc, s) => acc + s.time_on_site, 0) / state.sessions.length;
    
    const personaCounts = state.sessions.reduce((acc, s) => {
      acc[s.persona.type] = (acc[s.persona.type] || 0) + 1;
      return acc;
    }, {});
    
    const topPersona = Object.entries(personaCounts)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A';

    state.analytics = {
      totalSessions: state.sessions.length,
      conversionRate: Math.round((conversions / state.sessions.length) * 100),
      avgSessionTime: Math.round(avgTime),
      topPersona
    };

    UIRenderer.renderAnalytics();
  }
}

// Main Application Controller
class App {
  static async init() {
    console.log('Initializing Shopper Behavior Predictor...');
    
    // Setup event listeners
    document.getElementById('add-session-btn').addEventListener('click', () => {
      this.addSession();
    });
    
    document.getElementById('toggle-auto-btn').addEventListener('click', () => {
      this.toggleAutoGenerate();
    });
    
    // Initial session
    await this.addSession();
    
    // Start auto-generation
    this.startAutoGeneration();
  }

  static async addSession() {
    UIRenderer.showLoading();
    
    const sessionData = SessionGenerator.generate();
    const prediction = await APIClient.predict(sessionData);
    
    const session = {
      ...sessionData,
      ...prediction
    };
    
    state.sessions.unshift(session);
    if (state.sessions.length > 100) {
      state.sessions.pop();
    }
    
    UIRenderer.renderSession(session);
    AnalyticsCalculator.calculate();
    
    UIRenderer.hideLoading();
  }

  static startAutoGeneration() {
    setInterval(() => {
      if (state.autoGenerate) {
        this.addSession();
      }
    }, 5000);
  }

  static toggleAutoGenerate() {
    state.autoGenerate = !state.autoGenerate;
    const btn = document.getElementById('toggle-auto-btn');
    btn.textContent = state.autoGenerate ? 'Pause Auto-Gen' : 'Resume Auto-Gen';
    btn.classList.toggle('paused');
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  App.init();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { App, APIClient, SessionGenerator, UIRenderer };
}