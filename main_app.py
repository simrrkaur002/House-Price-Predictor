import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set Streamlit page config
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Load trained model (must match features used in training)
model = joblib.load("model.pkl")

# Load dataset for visualizations
df = pd.read_csv("train.csv")

# Define features used in training
features = [
    'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt',
    'LotArea', '1stFlrSF', 'FullBath', 'BedroomAbvGr', 'Fireplaces'
]

# Filter and clean dataset
df = df[features + ['SalePrice']].dropna()

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: white;
    }
    body {
        background-color: #0d1117;
        color: white;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🏡 House Price Predictor")
st.markdown("Use the sidebar to enter house features and get a price prediction.")

# --- Sidebar: User Inputs ---
st.sidebar.header("📋 Input House Features")
overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sqft)", 300, 5000, 1500)
garage_area = st.sidebar.number_input("Garage Area (sqft)", 0, 1500, 500)
total_bsmt_sf = st.sidebar.number_input("Basement Area (sqft)", 0, 3000, 800)
year_built = st.sidebar.number_input("Year Built", 1800, 2023, 2000)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", 500, 30000, 8500)
first_flr_sf = st.sidebar.number_input("1st Floor SF", 200, 3000, 1200)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
bedroom_abvgr = st.sidebar.slider("Bedrooms Above Ground", 0, 10, 3)
fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1)


# --- Predict Button ---
if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageArea": [garage_area],
        "TotalBsmtSF": [total_bsmt_sf],
        "YearBuilt": [year_built],
        "LotArea": [lot_area],
        "1stFlrSF": [first_flr_sf],
        "FullBath": [full_bath],
        "BedroomAbvGr": [bedroom_abvgr],
        "Fireplaces": [fireplaces]
    })

    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    predicted_price = model.predict(input_df)[0] 
    st.success(f"💰 Estimated House Price: **${predicted_price:,.2f}**")
    
# --- Visualizations Section ---
st.header("📊 Data Visualizations")

# Feature vs. SalePrice scatter plot
selected_feature = st.selectbox("Select Feature to Compare with SalePrice", features)
fig, ax = plt.subplots()
sns.scatterplot(data=df, x=selected_feature, y='SalePrice', ax=ax)
ax.set_title(f"{selected_feature} vs SalePrice")
st.pyplot(fig)

# Correlation Heatmap
if st.checkbox("Show Correlation Heatmap"):
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    corr = df[features + ['SalePrice']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
