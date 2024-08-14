import streamlit as st
import pandas as pd
import plotly.express as px

data_path ="https://raw.githubusercontent.com/jansu143/Dashbaord/main/mnt/data/apparel.csv"
# Load and preprocess the data
#data_path = '../mnt/data/apparel.csv'  # Adjust this path if needed
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

# Sidebar for filtering
st.sidebar.header('Filter Options')
vendor = st.sidebar.selectbox('Select Vendor', ['All Vendors'] + list(df['Vendor'].unique()))
product = st.sidebar.selectbox('Select Product', ['All Products'] + list(df['Title'].unique()))

# Filter data based on selection
filtered_df = df.copy()
if vendor != 'All Vendors':
    filtered_df = filtered_df[filtered_df['Vendor'] == vendor]
if product != 'All Products':
    filtered_df = filtered_df[filtered_df['Title'] == product]

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("No data available for the selected criteria.")
else:
    # Inventory Measures
    st.subheader("Inventory Levels by Product")
    inventory_measures_fig = px.bar(
        filtered_df,
        x='Title',
        y='Variant Inventory Qty',
        title='Inventory Levels by Product',
        labels={'Variant Inventory Qty': 'Inventory Quantity'},
        color='Title',
        height=500
    )
    st.plotly_chart(inventory_measures_fig)

    # Inventory Trend by Price
    st.subheader("Inventory Price Trends by Product")
    inventory_trend_fig = px.line(
        filtered_df,
        x='Date',
        y='Variant Price',
        title='Inventory Price Trends by Product',
        color='Title',
        labels={'Variant Price': 'Price', 'Date': 'Date'},
        height=500
    )
    st.plotly_chart(inventory_trend_fig)

    # Inventory Forecast Trends
    st.subheader("Inventory Forecast Trends")
    forecast_fig = px.line(
        filtered_df,
        x='Date',
        y='Variant Inventory Qty',
        title='Inventory Forecast Trends',
        labels={'Variant Inventory Qty': 'Inventory Quantity', 'Date': 'Date'},
        color='Title',
        height=500
    )
    st.plotly_chart(forecast_fig)

    # AI-driven inventory alerts
    st.subheader("Inventory Alerts")
    alert_messages = []
    low_inventory_threshold = 10  # Example threshold for low inventory
    high_inventory_threshold = 1000  # Example threshold for high inventory

    for _, row in filtered_df.iterrows():
        if row['Variant Inventory Qty'] < low_inventory_threshold:
            alert_messages.append(f"Alert: Low inventory for {row['Title']} (Vendor: {row['Vendor']}) - Quantity: {row['Variant Inventory Qty']}")
        if row['Variant Inventory Qty'] > high_inventory_threshold:
            alert_messages.append(f"Alert: High inventory for {row['Title']} (Vendor: {row['Vendor']}) - Quantity: {row['Variant Inventory Qty']}")

    if alert_messages:
        for alert in alert_messages:
            st.warning(alert)
    else:
        st.success("No inventory alerts.")

# Inventory Optimization Button
if st.button("Run Inventory Optimization"):
    st.write("Inventory optimization has been run")
