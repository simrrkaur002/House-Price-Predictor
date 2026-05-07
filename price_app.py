#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import base64

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #0d1117;
        color: white;
    }
    .main {
        background-color: #0d1117;
        color: white;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏡 House Price Predictor")
st.markdown("Welcome! Enter the house details below:")

st.sidebar.header("📋 Feature Inputs")

overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sqft)", min_value=300, max_value=5000, value=1500)
garage_area = st.sidebar.number_input("Garage Area (sqft)", min_value=0, max_value=1500, value=500)
total_bsmt_sf = st.sidebar.number_input("Basement Area (sqft)", min_value=0, max_value=2000, value=800)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2023, value=2000)
lot_area = st.sidebar.number_input("Lot Area (sq ft)", 500, 30000, 8500)
first_flr_sf = st.sidebar.number_input("1st Floor SF", 200, 3000, 1200)
full_bath = st.sidebar.slider("Full Bathrooms", 0, 4, 2)
bedroom_abvgr = st.sidebar.slider("Bedrooms Above Ground", 0, 10, 3)
fireplaces = st.sidebar.slider("Fireplaces", 0, 3, 1)

if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageArea": [garage_area],
        "TotalBsmtSF": [total_bsmt_sf],
        "YearBuilt": [year_built]
    })

    model = joblib.load("model.pkl")
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated House Price: ${round(prediction, 2)}")

df = pd.read_csv("train.csv")
features = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'YearBuilt']
df = df[features + ['SalePrice']].dropna()

X = df[features]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
