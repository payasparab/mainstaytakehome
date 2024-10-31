import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(layout="wide", page_title="OpenDoor Performance Dashboard")

# Reference to calculate_performance_metrics function from original notebook
# Lines 120-148 from mainstay_takehome.ipynb

# Reference to plot_performance_metrics function from original notebook
# Lines 154-229 from mainstay_takehome.ipynb

def load_data():
    """Load and preprocess the data"""
    data = pd.read_csv('data.csv')
    data['date'] = pd.to_datetime(data['date'])
    data['mls_listings'] = data.mls_listings.str.replace(',', '').astype('int')
    return data

def main():
    st.title("OpenDoor Performance Dashboard")
    
    # Load data
    data = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Price band filter
    price_bands = ['All'] + list(data['price_band'].unique())
    selected_price_bands = st.sidebar.multiselect(
        "Select Price Bands",
        price_bands,
        default=['All']
    )
    
    # Zip code filter
    zip_codes = ['All'] + list(data['zip_code'].unique())
    selected_zip_codes = st.sidebar.multiselect(
        "Select Zip Codes",
        zip_codes,
        default=['All']
    )
    
    # Filter data based on selections
    filtered_data = data.copy()
    if 'All' not in selected_price_bands:
        filtered_data = filtered_data[filtered_data['price_band'].isin(selected_price_bands)]
    if 'All' not in selected_zip_codes:
        filtered_data = filtered_data[filtered_data['zip_code'].isin(selected_zip_codes)]
    
    # Aggregate data
    numeric_cols = ['mls_listings', 'od_listings', 'mls_contracts', 'od_contracts', 'od_home_visits']
    agg_data = filtered_data.groupby('date')[numeric_cols].sum().reset_index()
    
    # Calculate performance metrics
    metrics_data = calculate_performance_metrics(agg_data)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Raw Data", "Time Series"])
    
    with tab1:
        st.header("Performance Metrics")
        fig = plot_performance_metrics(metrics_data, monthly=True)
        st.pyplot(fig)
    
    with tab2:
        st.header("Raw Data")
        st.dataframe(filtered_data)
    
    with tab3:
        st.header("Time Series Analysis")
        
        # Select metrics to display
        metrics = st.multiselect(
            "Select metrics to display",
            numeric_cols,
            default=['mls_listings', 'od_listings']
        )
        
        # Create time series plot
        fig, ax = plt.subplots(figsize=(12, 6))
        for metric in metrics:
            ax.plot(agg_data['date'], agg_data[metric], label=metric)
        
        ax.set_title("Time Series Analysis")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)

if __name__ == "__main__":
    main()