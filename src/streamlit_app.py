"""
Streamlit dashboard for Smart Shopper AI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Smart Shopper AI Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Constants
API_URL = "http://localhost:8000"
CACHE_TTL = 300  # 5 minutes

# Helper functions
@st.cache_data(ttl=CACHE_TTL)
def fetch_predictions(start_date, end_date):
    """Fetch prediction data from API."""
    try:
        response = requests.get(
            f"{API_URL}/predictions",
            params={"start_date": start_date, "end_date": end_date}
        )
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def fetch_personas():
    """Fetch persona profiles from API."""
    try:
        response = requests.get(f"{API_URL}/personas")
        return response.json()
    except Exception as e:
        st.error(f"Error fetching personas: {str(e)}")
        return {}

def create_metrics_cards(df):
    """Display key metrics in cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Purchase Probability",
            f"{df['purchase_probability'].mean():.2%}",
            f"{df['purchase_probability'].mean() - 0.5:.2%}"
        )
    
    with col2:
        conversion_rate = (df['actual_purchase'] == True).mean()
        st.metric(
            "Conversion Rate",
            f"{conversion_rate:.2%}",
            f"{conversion_rate - 0.15:.2%}"
        )
    
    with col3:
        incentive_rate = (df['incentive_accepted'] == True).mean()
        st.metric(
            "Incentive Acceptance Rate",
            f"{incentive_rate:.2%}",
            f"{incentive_rate - 0.25:.2%}"
        )
    
    with col4:
        active_visitors = len(df['session_id'].unique())
        st.metric(
            "Active Visitors",
            f"{active_visitors:,}",
            f"{active_visitors - 1000:,}"
        )

def plot_purchase_probability_distribution(df):
    """Plot distribution of purchase probabilities."""
    fig = px.histogram(
        df,
        x='purchase_probability',
        nbins=50,
        title='Distribution of Purchase Probabilities',
        labels={'purchase_probability': 'Purchase Probability'},
        color_discrete_sequence=['#3366cc']
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_persona_distribution(df):
    """Plot distribution of shopper personas."""
    persona_counts = df['predicted_persona'].value_counts()
    fig = px.pie(
        values=persona_counts.values,
        names=persona_counts.index,
        title='Distribution of Shopper Personas',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_incentive_effectiveness(df):
    """Plot effectiveness of different incentive types."""
    incentive_stats = df.groupby('recommended_incentive').agg({
        'incentive_accepted': 'mean',
        'session_id': 'count'
    }).reset_index()
    
    fig = px.bar(
        incentive_stats,
        x='recommended_incentive',
        y='incentive_accepted',
        title='Incentive Effectiveness by Type',
        labels={
            'recommended_incentive': 'Incentive Type',
            'incentive_accepted': 'Acceptance Rate'
        },
        color='session_id',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

# Main dashboard
def main():
    # Title and description
    st.title("🛒 Smart Shopper AI Dashboard")
    st.markdown("""
        Monitor real-time shopper behavior predictions and incentive recommendations.
        Use the filters below to analyze different time periods and segments.
    """)
    
    # Date filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now().date() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now().date()
        )
    
    # Fetch data
    df = fetch_predictions(start_date, end_date)
    if df.empty:
        st.warning("No data available for the selected period.")
        return
    
    # Display metrics
    st.subheader("📊 Key Metrics")
    create_metrics_cards(df)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        plot_purchase_probability_distribution(df)
        
    with col2:
        plot_persona_distribution(df)
    
    st.subheader("💰 Incentive Analysis")
    plot_incentive_effectiveness(df)
    
    # Detailed data view
    st.subheader("🔍 Detailed Prediction Data")
    st.dataframe(
        df.sort_values('created_at', ascending=False)
        .head(100)
        .style.format({
            'purchase_probability': '{:.2%}',
            'created_at': '{:%Y-%m-%d %H:%M:%S}'
        })
    )

if __name__ == "__main__":
    main()