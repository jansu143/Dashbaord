import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime

# Load and preprocess the data
data_path ="https://raw.githubusercontent.com/jansu143/Dashbaord/main/mnt/data/apparel.csv"
#data_path = '../mnt/data/apparel.csv'  # Adjust the path as necessary
df = pd.read_csv(data_path)

# Required columns based on provided details
required_columns = [
    'Handle', 'Title', 'Vendor', 'Variant Inventory Qty', 'Variant Price', 'Image Src'
]

# Verify all required columns are present
for col in required_columns:
    if col not in df.columns:
        st.error(f"Column '{col}' not found in the dataset")
        st.stop()

# Clean the data by removing any rows with null values in the 'Vendor' or 'Title' columns
df = df.dropna(subset=['Vendor', 'Title'])

# Simulate a 'Date' column for inventory trends
df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# Sidebar for filtering and customization
st.sidebar.header('Filter Options')
vendor = st.sidebar.selectbox('Select Vendor', ['All Vendors'] + list(df['Vendor'].unique()))
product = st.sidebar.selectbox('Select Product', ['All Products'] + list(df['Title'].unique()))
date_range = st.sidebar.date_input("Select Date Range", [datetime(2023, 1, 1), datetime(2023, 12, 31)])
low_inventory_threshold = st.sidebar.number_input('Low Inventory Threshold', min_value=1, max_value=500, value=10)
high_inventory_threshold = st.sidebar.number_input('High Inventory Threshold', min_value=500, max_value=5000, value=1000)
chart_type = st.sidebar.selectbox('Select Chart Type for Inventory Measures', ['Bar Chart', 'Line Chart'])

# Filter data based on selection
filtered_df = df.copy()
if vendor != 'All Vendors':
    filtered_df = filtered_df[filtered_df['Vendor'] == vendor]
if product != 'All Products':
    filtered_df = filtered_df[filtered_df['Title'] == product]
filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(date_range[0])) & (filtered_df['Date'] <= pd.to_datetime(date_range[1]))]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("No data available for the selected criteria.")
else:
    # Inventory Measures
    st.subheader("Inventory Measures")
    if chart_type == 'Bar Chart':
        inventory_measures_fig = px.bar(filtered_df, x='Title', y='Variant Inventory Qty', title='Inventory Measures')
    else:
        inventory_measures_fig = px.line(filtered_df, x='Title', y='Variant Inventory Qty', title='Inventory Measures')
    st.plotly_chart(inventory_measures_fig)

    # Inventory Trend
    st.subheader("Inventory Trend")
    inventory_trend_fig = px.line(filtered_df, x='Date', y='Variant Price', title='Inventory Trend')
    st.plotly_chart(inventory_trend_fig)

    # Predictive Analytics using Prophet
    st.subheader("Inventory Forecast using Predictive Analytics")
    forecast_period = st.sidebar.slider("Select Forecast Period (days)", 1, 365, 30)

    # Prepare data for Prophet
    prophet_df = filtered_df[['Date', 'Variant Inventory Qty']].rename(columns={'Date': 'ds', 'Variant Inventory Qty': 'y'})

    # Initialize and train the Prophet model
    model = Prophet()
    model.fit(prophet_df)

    # Create future dates dataframe
    future = model.make_future_dataframe(periods=forecast_period)
    
    # Predict future values
    forecast = model.predict(future)

    # Plot the forecast
    forecast_fig = px.line(forecast, x='ds', y='yhat', title=f'Inventory Forecast for next {forecast_period} days')
    forecast_fig.add_scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='markers', name='Actual')
    st.plotly_chart(forecast_fig)

    # AI-driven inventory alerts with interactive elements
    st.subheader("Inventory Alerts")
    alert_count = 0
    for _, row in filtered_df.iterrows():
        if row['Variant Inventory Qty'] < low_inventory_threshold:
            st.error(f"⚠️ Low inventory: {row['Title']} (Vendor: {row['Vendor']}) - Quantity: {row['Variant Inventory Qty']}")
            alert_count += 1
        elif row['Variant Inventory Qty'] > high_inventory_threshold:
            st.warning(f"⚠️ High inventory: {row['Title']} (Vendor: {row['Vendor']}) - Quantity: {row['Variant Inventory Qty']}")
            alert_count += 1

    if alert_count == 0:
        st.success("✅ No inventory alerts")

# Inventory Optimization Button
if st.button("Run Inventory Optimization"):
    st.write("Inventory optimization has been run.")

# Download filtered data as CSV
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_inventory_data.csv',
    mime='text/csv'
)
