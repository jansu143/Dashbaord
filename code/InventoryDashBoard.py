import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# Load the data
@st.cache_data
def inventory_dashboard():
    sales_data_path = '../mnt//data/apparel.csv'  # Update with your actual dataset path
    df = pd.read_csv(sales_data_path)
    return df

# Streamlit App
st.title("Inventory Optimization Dashboard Based on Demand")

# Load data
sales_df = inventory_dashboard()

# Display some basic information about the dataset
st.subheader("Data Overview")
st.write("Dimensions: ", sales_df.shape)
st.write("Missing values:", sales_df.isnull().sum())
st.dataframe(sales_df.head())

# Sidebar for filtering
st.sidebar.header('Filter Options')
vendor = st.sidebar.selectbox('Select Vendor', sales_df['Vendor'].dropna().unique())
product_type = st.sidebar.selectbox('Select Product Type', sales_df['Handle'].dropna().unique())

# Filtered data based on selection
filtered_sales = sales_df[(sales_df['Vendor'] == vendor) & (sales_df['Handle'] == product_type)]

# Check for missing values or insufficient data
if filtered_sales.empty or len(filtered_sales.dropna()) < 2:
    st.error("Not enough data available for the selected options. Please select another vendor or product type.")
else:
    # Simulating a date column as an example
    filtered_sales['Date'] = pd.date_range(start='2022-01-01', periods=len(filtered_sales), freq='D')
    filtered_sales = filtered_sales.rename(columns={'Date': 'ds', 'Variant Inventory Qty': 'y'})
    filtered_sales['ds'] = pd.to_datetime(filtered_sales['ds'])

    # Demand Forecasting using Prophet
    st.subheader("Demand Forecasting")
    model = Prophet()

    # Handle potential NaN values before fitting
    filtered_sales = filtered_sales.dropna(subset=['y'])

    model.fit(filtered_sales[['ds', 'y']])

    # Make future predictions
    future = model.make_future_dataframe(periods=30)  # Forecasting next 30 days
    forecast = model.predict(future)

    # Display forecast
    fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Demand')
    st.plotly_chart(fig_forecast)

    # Inventory Optimization
    st.subheader("Inventory Optimization Based on Demand")

    # Calculate Optimal Inventory
    lead_time_days = st.number_input("Enter Lead Time (days)", min_value=1, max_value=60, value=7)
    safety_stock_multiplier = st.number_input("Enter Safety Stock Multiplier", min_value=1.0, max_value=3.0, value=1.5)

    # Calculate Demand during Lead Time
    demand_during_lead_time = forecast.iloc[-lead_time_days:]['yhat'].sum()
    safety_stock = safety_stock_multiplier * forecast['yhat'].std()
    optimal_inventory_level = demand_during_lead_time + safety_stock

    # Display Results
    st.write(f"Optimal Inventory Level for {product_type} from {vendor}: {optimal_inventory_level:.2f} units")

    # Comparison with Current Inventory
    current_inventory = st.number_input("Enter Current Inventory Level", min_value=0)
    inventory_gap = optimal_inventory_level - current_inventory

    if inventory_gap > 0:
        st.success(f"Increase inventory by {inventory_gap:.2f} units.")
    else:
        st.warning(f"Reduce inventory by {abs(inventory_gap):.2f} units.")
