# app/streamlit_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_prep import prepare
from eda_plots import (
    price_distribution,
    price_per_sqft_by_city,
    correlation_heatmap,
    bhk_distribution_by_city
)



st.set_page_config(layout="wide", page_title="Real Estate Investment Advisor")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_data
def prep_and_train(df):
    df_clean, city_medians = prepare(df.copy())
    # Baseline regression: predict current Price_in_Lakhs from features (so we can estimate growth with residual)
    feat_cols = ['Size_in_SqFt','Price_per_SqFt','BHK','Age_of_Property','Amenities_count','Nearby_Schools','Parking_flag','Ready_to_Move']
    X = df_clean[feat_cols].fillna(0)
    y = df_clean['Price_in_Lakhs']
    # simple holdout
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    # compute city-level median annual growth approximated as residual / current price over 5 years if data had history
    # fallback: default growth
    return df_clean, city_medians, reg, feat_cols

def fixed_growth_projection(current_price_lakhs, annual_rate=0.08, years=5):
    return current_price_lakhs * ((1 + annual_rate) ** years)

# --- UI ---
st.title("🏠 Real Estate Investment Advisor (Data Prep + EDA + App)")
st.markdown("Load your `india_housing_prices.csv` and explore data, then evaluate a property with rule-based investment signal and 5-year price forecast.")

# Sidebar — data load & EDA toggle
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload india_housing_prices.csv", type=['csv'])
if uploaded:
    df_raw = load_data(uploaded)
else:
    st.sidebar.info("Or run with the sample dataset in `data/` and set path below.")
    sample_path = st.sidebar.text_input("Local CSV path (if not uploading)", value="../data/india_housing_prices.csv")
    try:
        df_raw = load_data(sample_path)
    except Exception:
        st.sidebar.warning("No dataset loaded yet. Upload a CSV or set correct path.")
        df_raw = None

if df_raw is None:
    st.stop()

# Prepare data and baseline model
with st.spinner("Preparing data and training baseline..."):
    df_clean, city_medians, baseline_reg, feat_cols = prep_and_train(df_raw)

st.sidebar.success("Data prepared")

# Show dataset summary
st.subheader("Dataset snapshot")
st.dataframe(df_clean.head(50))

# EDA section
st.subheader("Exploratory Data Analysis")
col1, col2 = st.columns([1,1])
with col1:
    st.plotly_chart(price_distribution(df_clean), use_container_width=True)
with col2:
    st.plotly_chart(price_per_sqft_by_city(df_clean, top_n=15), use_container_width=True)

with st.expander("Correlation heatmap"):
    st.plotly_chart(correlation_heatmap(df_clean), use_container_width=True)

# City specific EDA
city_choice = st.selectbox("Select city for BHK distribution", options=sorted(df_clean['City'].unique()))
if city_choice:
    st.plotly_chart(bhk_distribution_by_city(df_clean, city_choice), use_container_width=True)

# Investment evaluation form
st.subheader("Evaluate a Property")
with st.form("eval"):
    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("City", options=sorted(df_clean['City'].unique()), index=0)
        locality = st.text_input("Locality (optional)")
        property_type = st.selectbox("Property Type", options=sorted(df_clean['Property_Type'].fillna('Unknown').unique()))
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    with col2:
        sqft = st.number_input("Size (SqFt)", min_value=100, max_value=20000, value=800)
        price_lakhs = st.number_input("Current Price (lakhs)", min_value=0.1, value=50.0, step=0.1)
        furnished = st.selectbox("Furnished Status", options=sorted(df_clean['Furnished_Status'].fillna('Unknown').unique()))
    with col3:
        amenities = st.text_input("Amenities (comma separated)")
        nearby_schools = st.number_input("Nearby Schools (count)", min_value=0, value=1)
        parking = st.checkbox("Has parking?")
    submitted = st.form_submit_button("Evaluate Property")

if submitted:
    # build input row like training features
    input_row = {
        'City': city,
        'Size_in_SqFt': sqft,
        'Price_in_Lakhs': price_lakhs,
        'Price_per_SqFt': (price_lakhs*100000)/sqft if sqft>0 else 0,
        'BHK': int(bhk),
        'Amenities_count': len([a for a in amenities.split(',') if a.strip()]) if amenities else 0,
        'Ready_to_Move': 1,  # assume available
        'Parking_flag': int(parking),
        'Age_of_Property': df_clean['Age_of_Property'].median(),  # user can extend form for Year_Built
        'Nearby_Schools': nearby_schools
    }
    input_df = pd.DataFrame([input_row])
    # Rule-based Good Investment
    # Recompute city median
    city_med = city_medians.get(city, df_clean['Price_per_SqFt'].median())
    score = 0
    if input_row['Price_per_SqFt'] <= city_med: score += 1
    if input_row['BHK'] >= 3: score += 1
    if input_row['Ready_to_Move'] == 1: score += 1
    if input_row['Amenities_count'] >= 3: score += 1
    if input_row['Parking_flag'] == 1: score += 1
    is_good = score >= 3

    # Regression forecasts
    # Method 1: fixed-growth
    default_rate = st.sidebar.slider("Default annual growth rate (%)", 1, 20, 8)
    proj_fixed = fixed_growth_projection(price_lakhs, annual_rate=default_rate/100, years=5)

    # Method 2: baseline linear regression trained on dataset -> predict current price and scale by residual-based growth heuristic
    X_input = input_df[[c for c in feat_cols if c in input_df.columns]].fillna(0)
    pred_current = baseline_reg.predict(X_input)[0]  # predicted current price in lakhs
    # simple heuristic: if asking price is lower than model predicted, expect above-average growth
    rel = (pred_current - price_lakhs) / pred_current if pred_current else 0
    # convert relative undervaluation into extra growth factor (heuristic)
    bonus_rate = max(min(rel * 0.05, 0.05), -0.03)  # +/- adjustment within reasonable bounds
    effective_rate = default_rate/100 + bonus_rate
    proj_baseline = fixed_growth_projection(price_lakhs, annual_rate=effective_rate, years=5)

    # display results
    st.markdown("### Recommendation")
    st.metric("Good Investment (rule-based)", "YES ✅" if is_good else "NO ❌", delta=f"Score = {score}/5")
    st.markdown("### 5-Year Price Forecast")
    st.write(f"- Fixed growth ({default_rate}% p.a.): **{proj_fixed:.2f} lakhs**")
    st.write(f"- Baseline-adjusted (data heuristic): **{proj_baseline:.2f} lakhs** (effective annual rate ≈ {effective_rate*100:.2f}%)")

    # show short explanation and feature influence (from linear model coefficients)
    coefs = pd.Series(baseline_reg.coef_, index=feat_cols).sort_values(key=abs, ascending=False)
    st.subheader("Baseline model feature influence (coefficients)")
    st.table(coefs.reset_index().rename(columns={'index':'feature',0:'coefficient'}).head(10))

    # show relevant comparatives: city median price per sqft
    st.write(f"City median Price/SqFt for {city}: {city_med:.2f}")
    st.write("Notes: This app uses interpretable rules + a baseline linear model trained on your dataset. For production you can replace baseline with XGBoost or an MLflow-tracked model.")

# Footer
st.markdown("---")
st.caption("Built for the Real Estate Investment Advisor project — data prep, EDA and Streamlit delivery. Adjust defaults and extend with real ML if desired.")
