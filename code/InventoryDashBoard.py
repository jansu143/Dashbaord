import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
def inventory_dashboard():
# Load your inventory data
    inventory_data_path = '../mnt/data/apparel.csv'
    data_path = '../mnt/data/AmazonSaleReport.csv'  # Replace with actual path
    inventory_df = pd.read_csv(inventory_data_path)

    # Sidebar for filtering
    st.sidebar.header('Filter Options')
    region = st.sidebar.selectbox('Select Region', inventory_df['Region'].unique())
    date_range = st.sidebar.slider('Select Date Range', value=[inventory_df['Date'].min(), inventory_df['Date'].max()])

    # Filtered data based on selection
    filtered_inventory = inventory_df[(inventory_df['Region'] == region) & 
                                    (inventory_df['Date'].between(date_range[0], date_range[1]))]

    # Inventory Levels Over Time
    st.subheader("Inventory Levels Over Time")
    inventory_trend = filtered_inventory.groupby(['Date', 'Product'])['Inventory'].sum().reset_index()
    fig_inventory_trend = px.bar(inventory_trend, x='Date', y='Inventory', color='Product', title='Inventory Levels Over Time')
    st.plotly_chart(fig_inventory_trend)

    # Demand Forecasting (Example with Simulated Data)
    st.subheader("Demand Forecasting")
    forecast_period = st.slider('Forecast Period (weeks)', min_value=1, max_value=12, value=4)
    forecast_df = filtered_inventory.groupby(['Date', 'Product'])['Demand'].sum().reset_index()

    # Simulate a simple linear forecast (replace with actual model)
    forecast_df['Forecasted_Demand'] = forecast_df['Demand'].rolling(window=forecast_period).mean().shift(-forecast_period)
    fig_demand_forecast = px.line(forecast_df, x='Date', y='Forecasted_Demand', color='Product', title='Forecasted Demand')
    st.plotly_chart(fig_demand_forecast)

    # Inventory Optimization (Simple Example)
    st.subheader("Inventory Optimization")
    optimization_goal = st.selectbox('Optimization Goal', ['Minimize Cost', 'Maximize Service Level'])

    # Simulate Optimization Results (replace with actual optimization model)
    optimization_results = filtered_inventory.groupby(['Date', 'Region'])['Inventory'].sum().reset_index()
    optimization_results['Optimized_Inventory'] = optimization_results['Inventory'] * np.random.uniform(0.9, 1.1, len(optimization_results))

    fig_optimization = px.bar(optimization_results, x='Date', y='Optimized_Inventory', color='Region', title='Optimized Inventory Allocation')
    st.plotly_chart(fig_optimization)

    # Display Optimization Results
    st.write(optimization_results[['Date', 'Region', 'Optimized_Inventory']])
