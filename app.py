import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# Set page config
st.set_page_config(layout="wide", page_title="OpenDoor Performance Dashboard")

# Set numeric columns for aggregation
numeric_cols = ['mls_listings', 'mls_contracts', 'od_listings', 'od_contracts', 'od_home_visits']

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
    Plot only the selected performance metrics using Plotly.
    """
    # Set up resampling based on selected period
    if resample_period == 'monthly':
        plot_data = data_xs.groupby(data_xs['date'].dt.to_period('M')).mean()
        plot_data.index = plot_data.index.astype(str)
    elif resample_period == 'weekly':
        plot_data = data_xs.groupby(data_xs['date'].dt.to_period('W')).mean()
        plot_data.index = plot_data.index.astype(str)
    else:  # daily
        plot_data = data_xs.set_index('date')

    # Create figure
    fig = go.Figure()
    
    # Add traces for each metric
    for chart_name in selected_charts:
        metrics = chart_options[chart_name]
        for metric in metrics:
            if metric in plot_data.columns:  # Only plot if column exists
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data[metric],
                        name=metric,
                        mode='lines+markers',
                        hovertemplate=f"{metric}: %{{y:.3f}}<br>Date: %{{x}}<extra></extra>"
                    )
                )
    
    # Add reference line for conversion vs market chart if needed
    if "Contract Conversion vs Market" in selected_charts:
        fig.add_hline(y=1, line_dash="dot", line_color="black", opacity=0.5)

    # Update layout
    fig.update_layout(
        title=f"{', '.join(selected_charts)} Over Time ({resample_period.capitalize()})",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        showlegend=True,
        height=500,
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        margin=dict(r=150)
    )

    return fig

def main():
    st.title("OpenDoor Performance Dashboard")
    
    # Load data
    data = load_data()
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Calculated Performance Metrics", "Time Series Raw Metrics", "Downloadable Raw Data"])
    
    # Create controls first
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
            
            # Show checkbox for price bands (always available)
            show_separate_price_bands = st.checkbox('Show separate lines for each price band', key='price_bands_checkbox')
            
            # Zip code filter
            zip_codes = ['All'] + list(data['zip_code'].unique())
            selected_zip_codes = st.multiselect(
                "Select Zip Codes",
                zip_codes,
                default=['All'],
                key="zip_codes_select"
            )
            
            # Show checkbox for zip codes (always available)
            show_separate_zip_codes = st.checkbox('Show separate lines for each zip code', key='zip_codes_checkbox')
        
        with control_col3:
            # Date range filters
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Now filter the data based on selections
        filtered_data = data.copy()
        if 'All' not in selected_price_bands:
            filtered_data = filtered_data[filtered_data['price_band'].isin(selected_price_bands)]
        if 'All' not in selected_zip_codes:
            filtered_data = filtered_data[filtered_data['zip_code'].isin(selected_zip_codes)]
        
        filtered_data = filtered_data[
            (filtered_data['date'].dt.date >= start_date) & 
            (filtered_data['date'].dt.date <= end_date)
        ]
        
        # At the start of the data processing section, before any if conditions:
        agg_data = filtered_data.groupby('date')[numeric_cols].sum().reset_index()

        if show_separate_price_bands and show_separate_zip_codes:
            # Handle both cross-sections
            price_bands_to_use = data['price_band'].unique() if 'All' in selected_price_bands else selected_price_bands
            zip_codes_to_use = data['zip_code'].unique() if 'All' in selected_zip_codes else selected_zip_codes
            
            agg_data = filtered_data.groupby(['date', 'price_band', 'zip_code'])[numeric_cols].sum().reset_index()
            metrics_data = pd.DataFrame()
            
            for price_band in price_bands_to_use:
                for zip_code in zip_codes_to_use:
                    subset_data = calculate_performance_metrics(
                        agg_data[(agg_data['price_band'] == price_band) & 
                                (agg_data['zip_code'] == zip_code)]
                    )
                    # Only keep date and the metrics we want to plot
                    metrics_to_keep = ['date'] + [col for col in chart_options[selected_chart]]
                    subset_data = subset_data[metrics_to_keep].copy()
                    
                    # Rename columns with suffix
                    subset_data.columns = ['date' if col == 'date' else f'{col}_{price_band}_{zip_code}' 
                                         for col in subset_data.columns]
                    
                    if metrics_data.empty:
                        metrics_data = subset_data
                    else:
                        metrics_data = metrics_data.merge(subset_data, on='date')
            
            # Update chart_options for the selected chart
            chart_options[selected_chart] = [
                f'{metric}_{band}_{zip}' 
                for metric in chart_options[selected_chart]
                for band in price_bands_to_use 
                for zip in zip_codes_to_use
            ]
            
        elif show_separate_price_bands:
            price_bands_to_use = data['price_band'].unique() if 'All' in selected_price_bands else selected_price_bands
            metrics_data = pd.DataFrame()
            new_metric_names = []
            
            for price_band in price_bands_to_use:
                # First filter data for this price band
                price_band_subset = filtered_data[filtered_data['price_band'] == price_band]
                
                # Aggregate numeric columns only
                price_band_agg = price_band_subset.groupby('date')[numeric_cols].sum().reset_index()
                
                # Calculate performance metrics
                price_band_metrics = calculate_performance_metrics(price_band_agg)
                
                # Only keep date and relevant metrics
                base_metrics = chart_options[selected_chart]
                metrics_to_keep = ['date'] + base_metrics
                price_band_metrics = price_band_metrics[metrics_to_keep].copy()
                
                # Rename columns with suffix
                new_columns = ['date' if col == 'date' else f'{col}_{price_band}' 
                             for col in price_band_metrics.columns]
                price_band_metrics.columns = new_columns
                
                # Track new metric names
                new_metric_names.extend([col for col in new_columns if col != 'date'])
                
                # Merge with existing data
                if metrics_data.empty:
                    metrics_data = price_band_metrics
                else:
                    metrics_data = metrics_data.merge(price_band_metrics, on='date')
            
            # Update chart options after all columns are created
            chart_options[selected_chart] = new_metric_names

        elif show_separate_zip_codes:
            zip_codes_to_use = data['zip_code'].unique() if 'All' in selected_zip_codes else selected_zip_codes
            metrics_data = pd.DataFrame()
            new_metric_names = []  # Store the new column names
            
            for zip_code in zip_codes_to_use:
                # First filter data for this zip code
                zip_subset = filtered_data[filtered_data['zip_code'] == zip_code]
                
                # Aggregate the numeric columns
                zip_agg = zip_subset.groupby('date')[numeric_cols].sum().reset_index()
                
                # Calculate performance metrics
                zip_metrics = calculate_performance_metrics(zip_agg)
                
                # Only keep date and the metrics we want to plot
                base_metrics = chart_options[selected_chart]
                metrics_to_keep = ['date'] + base_metrics
                zip_metrics = zip_metrics[metrics_to_keep].copy()
                
                # Rename columns with suffix (except date)
                new_columns = []
                for col in zip_metrics.columns:
                    if col == 'date':
                        new_columns.append(col)
                    else:
                        new_name = f'{col}_{zip_code}'
                        new_columns.append(new_name)
                        new_metric_names.append(new_name)
                
                zip_metrics.columns = new_columns
                
                # Merge with existing data
                if metrics_data.empty:
                    metrics_data = zip_metrics
                else:
                    metrics_data = metrics_data.merge(zip_metrics, on='date')
            
            # Update chart options after all columns are created
            chart_options[selected_chart] = new_metric_names

        else:
            # Original aggregation code for no separate lines
            metrics_data = calculate_performance_metrics(agg_data)
        
        # Display current chart description with bold title and definition
        st.markdown(f"""
        **Currently Displayed Metric: {selected_chart}**  
        _{chart_descriptions[selected_chart]}_
        """)
        
        # Now display the chart
        if selected_chart:
            fig = plot_selected_metrics(
                metrics_data, 
                [selected_chart],
                chart_options, 
                resample_period=time_period.lower()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Download options
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                # Save figure as HTML
                buffer = io.StringIO()
                fig.write_html(buffer)
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="Download Interactive Chart (HTML)",
                    data=html_bytes,
                    file_name=f"{selected_chart}_{time_period.lower()}.html",
                    mime="text/html"
                )
            
            with download_col2:
                # Get the plotted data for CSV
                if time_period.lower() == 'monthly':
                    plot_data = metrics_data.groupby(metrics_data['date'].dt.to_period('M')).mean()
                elif time_period.lower() == 'weekly':
                    plot_data = metrics_data.groupby(metrics_data['date'].dt.to_period('W')).mean()
                else:  # daily
                    plot_data = metrics_data.set_index('date')
                
                csv = plot_data[chart_options[selected_chart]].to_csv()
                st.download_button(
                    label="Download Data as CSV", 
                    data=csv,
                    file_name=f"{selected_chart}_{time_period.lower()}.csv",
                    mime="text/csv"
                )
            
            with download_col3:
                # Save as static image using matplotlib
                plt.figure(figsize=(12,6))
                for metric in chart_options[selected_chart]:
                    # Convert Period index to string for matplotlib
                    x_values = [str(x) for x in plot_data.index]
                    plt.plot(x_values, plot_data[metric], label=metric, marker='o')
                plt.title(f"{selected_chart} Over Time ({time_period.capitalize()})")
                plt.xlabel("Date") 
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                
                # Save to bytes buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', bbox_inches='tight')
                plt.close()
                
                st.download_button(
                    label="Download as Image (PNG)",
                    data=img_buffer.getvalue(),
                    file_name=f"{selected_chart}_{time_period.lower()}.png",
                    mime="image/png"
                )
    with tab2:
        st.header("Time Series Analysis")
        
        # Add metrics description in expandable section
        with st.expander("Click here to understand data definitions"):
            st.markdown("""
            ### Field Definitions
            
            #### Geographic and Price Segmentation
            - **price_band**: Original list price bucketed below and above $200K
            - **zip_code**: A, B, C, and D denote 4 different, illustrative zips
            
            #### Market Activity Metrics
            - **mls_listings**: Total active listings on the market on any given day
            - **mls_contracts**: Resales contracts
            
            #### OpenDoor Activity Metrics
            - **od_listings**: Total Opendoor active listings on the market on any given day
            - **od_contracts**: Opendoor resales contracts
            - **od_home_visits**: Total home visits on all active listings
            """)
        
        # Create three columns for controls
        ts_col1, ts_col2, ts_col3 = st.columns(3)
        
        with ts_col1:
            # Single metric selector
            metric_descriptions = {
                'mls_listings': 'Total active listings on the market on any given day',
                'mls_contracts': 'Resales contracts',
                'od_listings': 'Total Opendoor active listings on the market on any given day',
                'od_contracts': 'Opendoor resales contracts',
                'od_home_visits': 'Total home visits on all active listings'
            }
            
            selected_metric = st.selectbox(
                "Select metric to display",
                options=numeric_cols,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            # Display current metric description
            st.markdown(f"""
            **Currently Displayed Metric:**  
            _{metric_descriptions.get(selected_metric, '')}_ 
            """)
            
            # Time period selector (matching Performance Metrics)
            time_period = st.selectbox(
                "Select time period",
                options=["Daily", "Weekly", "Monthly"],
                index=2,  # Default to Monthly
                key="ts_time_period"
            )
        
        with ts_col2:
            # Price band filter
            price_bands = ['All'] + list(data['price_band'].unique())
            selected_price_bands = st.multiselect(
                "Select Price Bands",
                price_bands,
                default=['All'],
                key="ts_price_bands_select"
            )
            
            # Show checkbox for price bands
            show_separate_price_bands = st.checkbox('Show separate lines for each price band', key='ts_price_bands_checkbox')
            
            # Zip code filter
            zip_codes = ['All'] + list(data['zip_code'].unique())
            selected_zip_codes = st.multiselect(
                "Select Zip Codes",
                zip_codes,
                default=['All'],
                key="ts_zip_codes_select"
            )
            
            # Show checkbox for zip codes
            show_separate_zip_codes = st.checkbox('Show separate lines for each zip code', key='ts_zip_codes_checkbox')
        
        with ts_col3:
            # Date range filters
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="ts_start_date")
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="ts_end_date")
        
        # Filter data based on selections
        ts_filtered_data = data.copy()
        if 'All' not in selected_price_bands:
            ts_filtered_data = ts_filtered_data[ts_filtered_data['price_band'].isin(selected_price_bands)]
        if 'All' not in selected_zip_codes:
            ts_filtered_data = ts_filtered_data[ts_filtered_data['zip_code'].isin(selected_zip_codes)]
        
        ts_filtered_data = ts_filtered_data[
            (ts_filtered_data['date'].dt.date >= start_date) & 
            (ts_filtered_data['date'].dt.date <= end_date)
        ]
        
        # Create figure based on selections
        fig = go.Figure()
        
        if show_separate_price_bands and show_separate_zip_codes:
            for price_band in (data['price_band'].unique() if 'All' in selected_price_bands else selected_price_bands):
                for zip_code in (data['zip_code'].unique() if 'All' in selected_zip_codes else selected_zip_codes):
                    subset = ts_filtered_data[
                        (ts_filtered_data['price_band'] == price_band) & 
                        (ts_filtered_data['zip_code'] == zip_code)
                    ]
                    
                    if time_period.lower() == 'monthly':
                        plot_data = subset.groupby(subset['date'].dt.to_period('M'))[selected_metric].mean()
                        plot_data.index = plot_data.index.astype(str)
                    elif time_period.lower() == 'weekly':
                        plot_data = subset.groupby(subset['date'].dt.to_period('W'))[selected_metric].mean()
                        plot_data.index = plot_data.index.astype(str)
                    else:  # daily
                        plot_data = subset.set_index('date')[selected_metric]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data.index,
                            y=plot_data,
                            name=f"{price_band} - {zip_code}",
                            mode='lines+markers',
                            hovertemplate=f"{selected_metric}: %{{y}}<br>Date: %{{x}}<extra></extra>"
                        )
                    )
        
        elif show_separate_price_bands:
            for price_band in (data['price_band'].unique() if 'All' in selected_price_bands else selected_price_bands):
                subset = ts_filtered_data[ts_filtered_data['price_band'] == price_band]
                
                if time_period.lower() == 'monthly':
                    plot_data = subset.groupby(subset['date'].dt.to_period('M'))[selected_metric].mean()
                    plot_data.index = plot_data.index.astype(str)
                elif time_period.lower() == 'weekly':
                    plot_data = subset.groupby(subset['date'].dt.to_period('W'))[selected_metric].mean()
                    plot_data.index = plot_data.index.astype(str)
                else:  # daily
                    plot_data = subset.set_index('date')[selected_metric]
                
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data,
                        name=price_band,
                        mode='lines+markers',
                        hovertemplate=f"{selected_metric}: %{{y}}<br>Date: %{{x}}<extra></extra>"
                    )
                )
        
        elif show_separate_zip_codes:
            for zip_code in (data['zip_code'].unique() if 'All' in selected_zip_codes else selected_zip_codes):
                subset = ts_filtered_data[ts_filtered_data['zip_code'] == zip_code]
                
                if time_period.lower() == 'monthly':
                    plot_data = subset.groupby(subset['date'].dt.to_period('M'))[selected_metric].mean()
                    plot_data.index = plot_data.index.astype(str)
                elif time_period.lower() == 'weekly':
                    plot_data = subset.groupby(subset['date'].dt.to_period('W'))[selected_metric].mean()
                    plot_data.index = plot_data.index.astype(str)
                else:  # daily
                    plot_data = subset.set_index('date')[selected_metric]
                
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.index,
                        y=plot_data,
                        name=zip_code,
                        mode='lines+markers',
                        hovertemplate=f"{selected_metric}: %{{y}}<br>Date: %{{x}}<extra></extra>"
                    )
                )
        
        else:
            if time_period.lower() == 'monthly':
                plot_data = ts_filtered_data.groupby(ts_filtered_data['date'].dt.to_period('M'))[selected_metric].mean()
                plot_data.index = plot_data.index.astype(str)
            elif time_period.lower() == 'weekly':
                plot_data = ts_filtered_data.groupby(ts_filtered_data['date'].dt.to_period('W'))[selected_metric].mean()
                plot_data.index = plot_data.index.astype(str)
            else:  # daily
                plot_data = ts_filtered_data.set_index('date')[selected_metric]
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data,
                    name=selected_metric,
                    mode='lines+markers',
                    hovertemplate=f"{selected_metric}: %{{y}}<br>Date: %{{x}}<extra></extra>"
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_metric.replace('_', ' ').title()} Over Time ({time_period})",
            xaxis_title="Date",
            yaxis_title=selected_metric.replace('_', ' ').title(),
            hovermode='x unified',
            showlegend=True,
            height=500,
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            ),
            margin=dict(r=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            # Save figure as HTML
            buffer = io.StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            st.download_button(
                label="Download Interactive Chart (HTML)",
                data=html_bytes,
                file_name=f"{selected_metric}_{time_period.lower()}.html",
                mime="text/html"
            )
        
        with download_col2:
            # Get the plotted data for CSV
            if time_period.lower() == 'monthly':
                plot_data = ts_filtered_data.groupby(ts_filtered_data['date'].dt.to_period('M'))[selected_metric].mean()
            elif time_period.lower() == 'weekly':
                plot_data = ts_filtered_data.groupby(ts_filtered_data['date'].dt.to_period('W'))[selected_metric].mean()
            else:  # daily
                plot_data = ts_filtered_data.set_index('date')[selected_metric]
            
            csv = plot_data.to_csv()
            st.download_button(
                label="Download Data as CSV", 
                data=csv,
                file_name=f"{selected_metric}_{time_period.lower()}.csv",
                mime="text/csv"
            )
        
        with download_col3:
            # Save as static image using matplotlib
            plt.figure(figsize=(12,6))
            plt.plot(plot_data.index.astype(str), plot_data.values, marker='o')
            plt.title(f"{selected_metric.replace('_', ' ').title()} Over Time ({time_period.capitalize()})")
            plt.xlabel("Date") 
            plt.ylabel(selected_metric.replace('_', ' ').title())
            plt.grid(True)
            plt.xticks(rotation=45)
            
            # Save to bytes buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            plt.close()
            
            st.download_button(
                label="Download as Image (PNG)",
                data=img_buffer.getvalue(),
                file_name=f"{selected_metric}_{time_period.lower()}.png",
                mime="image/png"
            )
    
    with tab3:
        st.header("Raw Data")
        
        # Create filter columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        # Categorical filters
        with filter_col1:
            st.write("### Categorical Filters")
            selected_price_bands = st.multiselect(
                "Filter Price Bands",
                options=['All'] + list(data['price_band'].unique()),
                default=['All'],
                key="raw_price_bands"
            )
            
            selected_zip_codes = st.multiselect(
                "Filter Zip Codes",
                options=['All'] + list(data['zip_code'].unique()),
                default=['All'],
                key="raw_zip_codes"
            )
        
        # Numeric filters with sliders
        with filter_col2:
            st.write("### Numeric Filters")
            numeric_filters = {}
            
            for col in numeric_cols:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                numeric_filters[col] = st.slider(
                    f"Filter {col.replace('_', ' ').title()}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"raw_{col}_slider"
                )
        
        # Date filters
        with filter_col3:
            st.write("### Date Filters")
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            start_date = st.date_input(
                "Start Date", 
                min_date, 
                min_value=min_date, 
                max_value=max_date,
                key="raw_start_date"
            )
            end_date = st.date_input(
                "End Date", 
                max_date, 
                min_value=min_date, 
                max_value=max_date,
                key="raw_end_date"
            )
        
        # Apply filters to create display data
        display_data = data.copy()
        
        # Apply categorical filters
        if 'All' not in selected_price_bands:
            display_data = display_data[display_data['price_band'].isin(selected_price_bands)]
        if 'All' not in selected_zip_codes:
            display_data = display_data[display_data['zip_code'].isin(selected_zip_codes)]
        
        # Apply numeric filters
        for col, (min_val, max_val) in numeric_filters.items():
            display_data = display_data[
                (display_data[col] >= min_val) & 
                (display_data[col] <= max_val)
            ]
        
        # Apply date filters
        display_data = display_data[
            (display_data['date'].dt.date >= start_date) & 
            (display_data['date'].dt.date <= end_date)
        ]
        
        # Display the filtered dataframe
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "date": st.column_config.DateColumn(
                    "Date",
                    format="YYYY-MM-DD",
                ),
                "price_band": st.column_config.TextColumn(
                    "Price Band",
                    help="Price band: above or below $200K",
                ),
                "zip_code": st.column_config.TextColumn(
                    "Zip Code",
                    help="Illustrative zip codes A, B, C, D",
                ),
            }
        )
        
        # Add download button for filtered data
        csv = display_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
        
        # Display summary statistics
        if st.checkbox("Show Summary Statistics"):
            st.write("### Summary Statistics")
            st.write(display_data.describe())

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