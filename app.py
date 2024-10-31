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

def calculate_performance_metrics(data_xs):
    """Calculate key performance metrics for Opendoor vs Market performance and add them to input dataframe
    
    Metrics added to dataframe:
    - mls_contract_to_listing_ratio: Conversion rate of MLS listings to contracts
    - od_contract_to_listing_ratio: Conversion rate of Opendoor listings to contracts
    - contract_conversion_vs_market: How Opendoor's conversion rate compares to market (>1 means better)
    - listings_od_market_share: Opendoor's share of total market listings
    - contracts_od_market_share: Opendoor's share of total market contracts
    - visits_to_listings_ratio: Average number of visits per Opendoor listing
    - visits_to_contracts_ratio: Number of visits needed per contract secured
    
    Returns:
        Original dataframe with additional calculated metric columns
    """
    # Conversion metrics - measure ability to convert listings to contracts
    data_xs['mls_contract_to_listing_ratio'] = data_xs.mls_contracts/data_xs.mls_listings
    data_xs['od_contract_to_listing_ratio'] = data_xs.od_contracts/data_xs.od_listings
    data_xs['contract_conversion_vs_market'] = data_xs.od_contract_to_listing_ratio/data_xs.mls_contract_to_listing_ratio
    
    # Market share metrics - measure Opendoor's presence in the market
    data_xs['listings_od_market_share'] = data_xs.od_listings/data_xs.mls_listings
    data_xs['contracts_od_market_share'] = data_xs.od_contracts/data_xs.mls_contracts
    
    # Visit metrics - measure efficiency of home visits in generating contracts
    data_xs['visits_to_listings_ratio'] = data_xs.od_home_visits/data_xs.od_listings
    data_xs['visits_to_contracts_ratio'] = data_xs.od_home_visits/data_xs.od_contracts
    
    return data_xs

def plot_performance_metrics(data_xs, monthly=True, title_flag=None):
    """
    Plot performance metrics in three separate graphs: conversion, visit, and market share metrics.
    
    Args:
        data_xs (pd.DataFrame): DataFrame containing the calculated metrics
        monthly (bool): If True, resample data to monthly averages. If False, use raw data
        title_flag (str, optional): Additional text to append to plot titles. Defaults to None.
    """
    # If monthly flag is True, resample to monthly averages
    if monthly:
        plot_data = data_xs.groupby(data_xs['date'].dt.to_period('M')).mean()
    else:
        plot_data = data_xs.set_index('date')
    
    # Prepare title suffix
    title_suffix = f" - {title_flag}" if title_flag else ""
    
    # Create six separate plots
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(12, 35))
    
    # Plot 1: Contract to Listing Ratios
    plot_data[['mls_contract_to_listing_ratio', 'od_contract_to_listing_ratio']].plot(ax=ax1)
    if monthly:
        for col in ['mls_contract_to_listing_ratio', 'od_contract_to_listing_ratio']:
            for idx, val in plot_data[col].items():
                ax1.annotate(f'{val:.3f}', (idx.to_timestamp(), val), textcoords="offset points", xytext=(0,10), ha='center')
    ax1.set_title(f'Contract to Listing Conversion Percent Over Time{title_suffix}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Contract Conversion vs Market
    plot_data['contract_conversion_vs_market'].plot(ax=ax2)
    if monthly:
        for idx, val in plot_data['contract_conversion_vs_market'].items():
            ax2.annotate(f'{val:.3f}', (idx.to_timestamp(), val), textcoords="offset points", xytext=(0,10), ha='center')
    ax2.axhline(y=1, linestyle=':', color='black')  # Add horizontal dotted line at y=1
    ax2.set_title(f'Contract Conversion vs Market Over Time{title_suffix}')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Market Share Metrics
    # Plot 3: Listings OD Market Share
    plot_data['listings_od_market_share'].plot(ax=ax3)
    if monthly:
        for idx, val in plot_data['listings_od_market_share'].items():
            ax3.annotate(f'{val:.3f}', (idx.to_timestamp(), val), textcoords="offset points", xytext=(0,10), ha='center')
    ax3.set_title(f'Listings OD Market Share Over Time{title_suffix}')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Contracts OD Market Share  
    plot_data['contracts_od_market_share'].plot(ax=ax4)
    if monthly:
        for idx, val in plot_data['contracts_od_market_share'].items():
            ax4.annotate(f'{val:.3f}', (idx.to_timestamp(), val), textcoords="offset points", xytext=(0,10), ha='center')
    ax4.set_title(f'Contracts OD Market Share Over Time{title_suffix}')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Visit Metrics
    # Plot 5: Visits to Listings Ratio
    plot_data['visits_to_listings_ratio'].plot(ax=ax5)
    if monthly:
        for idx, val in plot_data['visits_to_listings_ratio'].items():
            ax5.annotate(f'{val:.1f}', (idx.to_timestamp(), val), textcoords="offset points", xytext=(0,10), ha='center')
    ax5.set_title(f'Visits to Listings Ratio Over Time{title_suffix}')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 6: Visits to Contracts Ratio
    plot_data['visits_to_contracts_ratio'].plot(ax=ax6)
    if monthly:
        for idx, val in plot_data['visits_to_contracts_ratio'].items():
            ax6.annotate(f'{val:.1f}', (idx.to_timestamp(), val), textcoords="offset points", xytext=(0,10), ha='center')
    ax6.set_title(f'Visits to Contracts Ratio Over Time{title_suffix}')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    main()