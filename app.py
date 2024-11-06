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

def plot_selected_metrics(data_xs, selected_charts, chart_options, resample_period='monthly'):
    """
    Plot only the selected performance metrics.
    
    Args:
        data_xs (pd.DataFrame): DataFrame containing the calculated metrics
        selected_charts (list): List of chart names to display
        chart_options (dict): Dictionary mapping chart names to their metrics
        resample_period (str): Time period for resampling ('daily', 'weekly', or 'monthly')
    """
    # Set up resampling based on selected period
    if resample_period == 'monthly':
        plot_data = data_xs.groupby(data_xs['date'].dt.to_period('M')).mean()
    elif resample_period == 'weekly':
        plot_data = data_xs.groupby(data_xs['date'].dt.to_period('W')).mean()
    else:  # daily
        plot_data = data_xs.set_index('date')
    
    # Create subplot for each selected chart
    n_charts = len(selected_charts)
    fig, axes = plt.subplots(n_charts, 1, figsize=(12, 6 * n_charts))
    
    # Handle single subplot case
    if n_charts == 1:
        axes = [axes]
    
    for idx, (chart_name, ax) in enumerate(zip(selected_charts, axes)):
        metrics = chart_options[chart_name]
        
        # Plot the metrics
        plot_data[metrics].plot(ax=ax)
        
        # Add value annotations if not daily
        if resample_period != 'daily':
            for metric in metrics:
                for idx_date, val in plot_data[metric].items():
                    ax.annotate(
                        f'{val:.3f}', 
                        (idx_date.to_timestamp(), val), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center'
                    )
        
        # Add reference line for conversion vs market chart
        if chart_name == "Contract Conversion vs Market":
            ax.axhline(y=1, linestyle=':', color='black')
        
        ax.set_title(f'{chart_name} Over Time ({resample_period.capitalize()})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    st.title("OpenDoor Performance Dashboard")
    
    # Load data
    data = load_data()
    
    
    # Filter data based on selections
    filtered_data = data.copy()
    if 'All' not in selected_price_bands:
        filtered_data = filtered_data[filtered_data['price_band'].isin(selected_price_bands)]
    if 'All' not in selected_zip_codes:
        filtered_data = filtered_data[filtered_data['zip_code'].isin(selected_zip_codes)]
    
    # Apply date filters
    filtered_data = filtered_data[
        (filtered_data['date'].dt.date >= start_date) & 
        (filtered_data['date'].dt.date <= end_date)
    ]
    
    # Aggregate data
    numeric_cols = ['mls_listings', 'od_listings', 'mls_contracts', 'od_contracts', 'od_home_visits']
    agg_data = filtered_data.groupby('date')[numeric_cols].sum().reset_index()
    
    # Calculate performance metrics
    metrics_data = calculate_performance_metrics(agg_data)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", "Time Series", "Raw Data"])
    
    with tab1:
        st.header("Performance Metrics")
        
        # Add metrics description in expandable section
        with st.expander("Click here to understand definitions of metrics"):
            st.markdown("""
            ### Key Metrics Description
            
            #### Listing -> Contract Conversion Rates
            - **mls_contract_to_listing_ratio**: Conversion rate of market contracts/listings
            - **od_contract_to_listing_ratio**: Conversion rate of OD market contracts/listings
            - **contract_conversion_vs_market**: od conversion/mls_conversion  (>1 means OD performs better)
            
            #### Market Share
            - **listings_od_market_share**: OD's share of market listings
            - **contracts_od_market_share**: OD's share of market contracts
            
            #### Visits Relative to Listings and Conversion
            - **visits_to_listings_ratio**: Average number of visits per OpenDoor listing
            - **visits_to_contracts_ratio**: Number of visits needed per contract secured
            """)
        
        # Create three columns for controls
        control_col1, control_col2, control_col3 = st.columns(3)
        
        with control_col1:
            # Chart selector
            chart_options = {
                "Contract to Listing Ratios": ['mls_contract_to_listing_ratio', 'od_contract_to_listing_ratio'],
                "Contract Conversion vs Market": ['contract_conversion_vs_market'],
                "Listings Market Share": ['listings_od_market_share'],
                "Contracts Market Share": ['contracts_od_market_share'],
                "Visits to Listings Ratio": ['visits_to_listings_ratio'],
                "Visits to Contracts Ratio": ['visits_to_contracts_ratio']
            }
            
            # Dictionary of chart descriptions
            chart_descriptions = {
                "Contract to Listing Ratios": "Comparison of conversion rates between market (MLS) and OpenDoor listings to contracts",
                "Contract Conversion vs Market": "Ratio of OpenDoor's conversion rate to market conversion rate (>1 means OpenDoor performs better)",
                "Listings Market Share": "OpenDoor's share of total market listings",
                "Contracts Market Share": "OpenDoor's share of total market contracts",
                "Visits to Listings Ratio": "Average number of visits per OpenDoor listing",
                "Visits to Contracts Ratio": "Number of visits needed per contract secured"
            }
            
            selected_chart = st.selectbox(
                "Select chart to display",
                options=list(chart_options.keys()),
                index=0
            )
            
            # Time period selector
            time_period = st.selectbox(
                "Select time period",
                options=["Daily", "Weekly", "Monthly"],
                index=2  # Default to Monthly
            )
        
        with control_col2:
            # Price band filter
            price_bands = ['All'] + list(data['price_band'].unique())
            selected_price_bands = st.multiselect(
                "Select Price Bands",
                price_bands,
                default=['All'],
                key="price_bands_select"
            )
            
            # Zip code filter
            zip_codes = ['All'] + list(data['zip_code'].unique())
            selected_zip_codes = st.multiselect(
                "Select Zip Codes",
                zip_codes,
                default=['All'],
                key="zip_codes_select"
            )
        
        with control_col3:
            # Date range filters
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Display current chart description
        st.caption(f"📊 {chart_descriptions[selected_chart]}")
        
        if selected_chart:
            fig = plot_selected_metrics(
                metrics_data, 
                [selected_chart],
                chart_options, 
                resample_period=time_period.lower()
            )
            st.pyplot(fig)
            
            # Create columns for download buttons
            download_col1, download_col2 = st.columns(2)
            
            # Download chart as image
            with download_col1:
                # Save figure to bytes buffer
                from io import BytesIO
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight')
                btn = st.download_button(
                    label="Download Chart as PNG",
                    data=buf.getvalue(),
                    file_name=f"{selected_chart}_{time_period.lower()}.png",
                    mime="image/png"
                )
            
            # Download data as CSV
            with download_col2:
                # Get the plotted data
                if time_period.lower() == 'monthly':
                    plot_data = metrics_data.groupby(metrics_data['date'].dt.to_period('M')).mean()
                elif time_period.lower() == 'weekly':
                    plot_data = metrics_data.groupby(metrics_data['date'].dt.to_period('W')).mean()
                else:  # daily
                    plot_data = metrics_data.set_index('date')
                
                # Convert to CSV
                csv = plot_data[chart_options[selected_chart]].to_csv()
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name=f"{selected_chart}_{time_period.lower()}.csv",
                    mime="text/csv"
                )
    
    with tab2:
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
    
    with tab3:
        st.header("Raw Data")
        st.dataframe(filtered_data)

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