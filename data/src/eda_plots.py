# src/eda_plots.py
import plotly.express as px
import pandas as pd

def price_distribution(df):
    fig = px.histogram(df, x='Price_in_Lakhs', nbins=60, title='Property Price Distribution (lakhs)')
    return fig

def price_per_sqft_by_city(df, top_n=20):
    agg = df.groupby('City')['Price_per_SqFt'].median().sort_values(ascending=False).head(top_n)
    fig = px.bar(agg.reset_index(), x='City', y='Price_per_SqFt', title=f'Top {top_n} Cities by Median Price per SqFt')
    return fig

def bhk_distribution_by_city(df, city):
    sub = df[df['City']==city]
    fig = px.histogram(sub, x='BHK', title=f'BHK distribution in {city}')
    return fig

def correlation_heatmap(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title='Numeric Feature Correlation')
    return fig
