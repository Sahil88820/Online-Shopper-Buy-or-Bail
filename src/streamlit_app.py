import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json

API_URL = "http://127.0.0.1:8000/predict"

# ------------------- UI CONFIG ---------------------
st.set_page_config(
    page_title="Smart Shopper AI",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
body { background-color: #0b0e17 !important; }
.stApp { background-color: #0b0e17; }
.css-1d391kg { background-color: #131722; padding: 20px; border-radius: 12px; }
h1,h2,h3,label { color: white !important; }
.stMetric { background-color:#131722; padding:18px; border-radius:12px; border:1px solid #1f2a3c; }
input, select { border-radius: 8px !important; }
.sidebar .sidebar-content { background-color: #131722 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🛍️ Smart Shopper AI")
st.sidebar.write("Real-time shopper intelligence engine")
st.sidebar.markdown("---")
st.sidebar.write("⚙️ Powered by CatBoost + KMeans + Streamlit + FastAPI")

# ------------------- HEADER -----------------------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("""
    <h1>🧠 Smart Shopper AI</h1>
    <h3 style="color:#c6d5ff">Predict Purchase Intent • Persona • Incentive</h3>
    """, unsafe_allow_html=True)

with col2:
    st_lottie("https://assets9.lottiefiles.com/packages/lf20_u4yrau.json", height=120)

st.markdown("---")

# ------------------ INPUT FORM --------------------
with st.form("input_form"):
    st.subheader("🧾 Shopper Session Details")

    colA, colB, colC = st.columns(3)
    
    with colA:
        administrative = st.number_input("Administrative Pages", 0.0)
        informational = st.number_input("Informational Pages", 0.0)
        product_related = st.number_input("Product Related Pages", 0.0)

    with colB:
        administrative_duration = st.number_input("Admin Duration", 0.0)
        informational_duration = st.number_input("Informational Duration", 0.0)
        product_related_duration = st.number_input("Product Duration", 0.0)

    with colC:
        bounce_rates = st.number_input("Bounce Rate", 0.0, 1.0, 0.02)
        exit_rates = st.number_input("Exit Rate", 0.0, 1.0, 0.03)
        page_values = st.number_input("Page Value Score", 0.0)

    st.markdown("### 🌎 Visitor Metadata")

    col4, col5, col6, col7 = st.columns(4)
    month = col4.selectbox("Month", ["Feb","Mar","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    visitor_type = col5.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor"])
    weekend = col6.checkbox("Weekend Session")
    special_day = col7.number_input("Holiday Proximity (0-1)", 0.0, 1.0, 0.0)

    col8, col9, col10, col11 = st.columns(4)
    operating_systems = col8.number_input("OS", 1, 10, 2)
    browser = col9.number_input("Browser", 1, 10, 2)
    region = col10.number_input("Region", 1, 10, 1)
    traffic_type = col11.number_input("Traffic Source", 1, 20, 2)

    st.markdown("")
    submitted = st.form_submit_button("🚀 Predict Shopper Behavior", use_container_width=True)

# ---------------- SEND REQUEST & DISPLAY OUTPUT -----------------
if submitted:
    payload = {
        "administrative": administrative,
        "administrative_duration": administrative_duration,
        "informational": informational,
        "informational_duration": informational_duration,
        "product_related": product_related,
        "product_related_duration": product_related_duration,
        "bounce_rates": bounce_rates,
        "exit_rates": exit_rates,
        "page_values": page_values,
        "special_day": special_day,
        "month": month,
        "visitor_type": visitor_type,
        "weekend": weekend,
        "operating_systems": operating_systems,
        "browser": browser,
        "region": region,
        "traffic_type": traffic_type
    }
    st.write("📤 Sending to API:", payload)

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Buy Probability", f"{result['prediction']['buy_probability']*100:.2f}%")
        col2.metric("Persona", result["persona"]["persona_type"])
        col3.metric("Best Incentive", result["incentive"]["incentive_type"])

        st.success(f"🪄 Persona: **{result['persona']['persona_type']}** 🧍")
        st.info(f"🎁 Incentive to show: **{result['incentive']['incentive_type']}**")
        
        st.write("Debug JSON Response:", result)
    else:
        st.error("❌ API Error — check backend logs")
        st.write(response.text)
