import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Real Estate Valuation & Risk Assessment", layout="wide")

# --- LOAD MODEL & ENCODER ---
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("location_label_encoder.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("real_estate.csv")

# --- Load All Resources ---
try:
    model = load_model()
    le = load_encoder()
    data = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("📘 About This Project")
st.sidebar.markdown("""
**AI for Real Estate Valuation & Risk Assessment**
                    
This project uses machine learning to predict property prices and assess flood risks in different locations across India.

🔍 **Key Features**  
- Price prediction using Random Forest  
- Flood risk zone classification  
- Dynamic visualizations for price trends and risk levels  

🧰 **Technologies & Tools**  
- Python, Pandas, NumPy  
- Scikit-learn (ML Model)  
- Plotly (Interactive Graphs)  
- Streamlit (Web App Framework)  
- Label Encoding  
- Data from Kaggle & GIS Sources  
""")

# --- PAGE TITLE ---
st.title("🏡 Real Estate Price Prediction & Risk Analysis")
st.markdown("Enter the property details below to estimate the **price** and get a **location risk report**.")

# --- USER INPUT FORM ---
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    location = col1.selectbox("📍 Location", sorted(data['location'].dropna().unique()))
    bhk = col2.selectbox("🛏 BHK", [1, 2, 3, 4, 5])
    sqft = col3.number_input("📐 Total Sqft", min_value=300, max_value=10000, step=50, value=1000)

    col4, col5 = st.columns(2)
    bath = col4.selectbox("🛁 Bathrooms", [1, 2, 3, 4])
    balcony = col5.selectbox("🌿 Balcony", [0, 1, 2, 3])

    submitted = st.form_submit_button("🔍 Predict Price")

# --- PREDICTION OUTPUT ---
if submitted:
    try:
        # Calculate flood risk score from data
        flood_rows = data[data['location'] == location]
        if not flood_rows.empty:
            flood_percent = float(flood_rows['Corrected_Percent_Flooded_Area'].mean())
        else:
            flood_percent = 0.0  # fallback default

        risk_score = flood_percent / 100.0

        # Encode location
        location_encoded = le.transform([location])[0]

        # Prepare input array
        input_df = pd.DataFrame([{
            "location_encoded": location_encoded,
            "BHK": bhk,
            "bathrooms": bath,
            "balcony": balcony,
            "total_sqft": sqft,
            "risk_score": risk_score
        }])
        
        # Make prediction
        predicted_price = model.predict(input_df)[0]
        st.success(f"💰 Estimated Property Price: ₹ {round(predicted_price, 2)} Lakhs")

        # --- Risk Report ---
        st.subheader("⚠️ Risk Analysis")
        st.write(f"Flood Risk in **{location}**: **{round(flood_percent, 2)}%** area affected.")

        if flood_percent > 50:
            st.error("🚨 High Risk Zone")
        elif flood_percent > 25:
            st.warning("⚠️ Medium Risk Zone")
        else:
            st.success("✅ Low Risk Zone")

        # --- Enhanced Risk Chart ---
        risk_label = "High" if flood_percent > 50 else "Medium" if flood_percent > 25 else "Low"
        risk_color = "red" if risk_label == "High" else "orange" if risk_label == "Medium" else "green"

        chart_df = pd.DataFrame({
            "Risk Type": [f"{risk_label} Risk Zone"],
            "Flood Risk (%)": [flood_percent]
        })

        fig = px.bar(
            chart_df,
            x="Risk Type",
            y="Flood Risk (%)",
            color_discrete_sequence=[risk_color],
            text="Flood Risk (%)",
            title=f"📊 Flood Risk Zone Classification in {location}",
            height=400
        )

        fig.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )

        fig.update_layout(
            yaxis_range=[0, 100],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Price Trend Chart for Selected Location ---
        st.subheader("📈 Price Trend in Selected Location")

        location_df = data[data['location'] == location].copy()
        location_df = location_df.sort_values(by='price')

        if not location_df.empty:
            fig_price = px.line(
                location_df,
                x=location_df.index,
                y='price',
                title=f"💹 Price Flow in {location}",
                labels={'x': 'Properties (ordered by price)', 'price': 'Price (Lakhs)'},
                markers=True
            )

            fig_price.update_traces(line=dict(color="blue", width=2), marker=dict(size=6))
            fig_price.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14),
                height=400
            )

            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("ℹ️ Not enough data to show price trend for this location.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("🔧  AI for Real Estate Valuation & Risk Assessment ")

try:
    model = load_model()
    le = load_encoder()
    data = load_data()
    st.success("✅ Model and data loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

