# src/data_prep.py
import pandas as pd
import numpy as np
from datetime import datetime

CURRENT_YEAR = datetime.now().year

def compute_price_per_sqft(df):
    if 'Price_per_SqFt' not in df.columns or df['Price_per_SqFt'].isnull().all():
        df['Price_per_SqFt'] = (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt']
    return df

def clean_basics(df):
    # Drop exact duplicates
    df = df.drop_duplicates()
    # Remove rows with missing critical fields
    df = df.dropna(subset=['City','Size_in_SqFt','Price_in_Lakhs'], how='any')
    # Unify text fields
    for c in ['City','State','Property_Type','Furnished_Status','Availability_Status','Owner_Type']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str).str.strip()
    return df

def impute_numerics(df):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    return df

def feature_engineer(df):
    # Age_of_Property
    if 'Year_Built' in df.columns:
        df['Age_of_Property'] = df['Year_Built'].apply(
            lambda y: CURRENT_YEAR - y if pd.notnull(y) and y > 1800 else np.nan
        )
    if 'Age_of_Property' not in df.columns:
        df['Age_of_Property'] = np.nan
    df['Age_of_Property'] = df['Age_of_Property'].fillna(df['Age_of_Property'].median())

    # Amenities_count
    if 'Amenities' in df.columns:
        df['Amenities_count'] = df['Amenities'].fillna('').apply(
            lambda s: len([x for x in s.split(',') if x.strip()])
        )
    else:
        df['Amenities_count'] = 0

    # Ready_to_Move flag
    if 'Availability_Status' in df.columns:
        df['Ready_to_Move'] = (
            df['Availability_Status']
            .astype(str)
            .str.contains('Available|Ready', case=False, na=False)
            .astype(int)
        )
    else:
        df['Ready_to_Move'] = 0

    # Parking_flag
    if 'Parking_Space' in df.columns:
        df['Parking_flag'] = (
            df['Parking_Space']
            .astype(str)
            .str.lower()
            .isin(['yes', 'y', 'true', '1', 'available'])
            .astype(int)
        )
    else:
        df['Parking_flag'] = 0

    # Nearby counts -> fillna 0
    for c in ['Nearby_Schools','Nearby_Hospitals','Public_Transport_Accessibility']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        else:
            df[c] = 0

    # Price per sqft
    df = compute_price_per_sqft(df)

    # BHK numeric
    if 'BHK' in df.columns:
        df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce').fillna(1).astype(int)
    else:
        df['BHK'] = 1

    return df

def create_investment_label(df, score_threshold=3):
    # Compute city median price_per_sqft
    city_medians = df.groupby('City')['Price_per_SqFt'].median().to_dict()

    def label_row(r):
        score = 0
        city_med = city_medians.get(r['City'], r['Price_per_SqFt'])
        if r['Price_per_SqFt'] <= city_med: score += 1
        if r['BHK'] >= 3: score += 1
        if r['Ready_to_Move'] == 1: score += 1
        if r['Amenities_count'] >= 3: score += 1
        if r['Parking_flag'] == 1: score += 1
        return 1 if score >= score_threshold else 0

    df['Good_Investment'] = df.apply(label_row, axis=1)
    return df, city_medians

def prepare(df):
    df = clean_basics(df)
    df = impute_numerics(df)
    df = feature_engineer(df)
    df, city_medians = create_investment_label(df)
    return df, city_medians
