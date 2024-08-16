import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from datetime import datetime
import numpy as np
data_path ="https://raw.githubusercontent.com/jansu143/Dashbaord/main/mnt/data/apparel.csv"
# Load and preprocess the data
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

# Simulate External Market Data (e.g., Competitor Prices, Economic Indicators)
# Replace this section with actual data retrieval from an API or CSV
df['Competitor Price'] = df['Variant Price'] * np.random.uniform(0.8, 1.2, size=len(df))
df['Economic Indicator'] = np.random.uniform(100, 200, size=len(df))  # Simulated economic indicator

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
    st.warning("No data available for the selected criteria")
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

    # Display Competitor Price Comparison
    st.subheader("Competitor Price Comparison")
    competitor_price_fig = px.line(filtered_df, x='Date', y=['Variant Price', 'Competitor Price'], title='Your Price vs. Competitor Price')
    st.plotly_chart(competitor_price_fig)

    # Display Economic Indicator Trend
    st.subheader("Economic Indicator Trend")
    economic_indicator_fig = px.line(filtered_df, x='Date', y='Economic Indicator', title='Economic Indicator Trend')
    st.plotly_chart(economic_indicator_fig)

    # Weekly Forecasting and Stock Adjustment with External Data
    st.subheader("Weekly Inventory Forecast and Stock Adjustment (with External Data)")
    forecast_weeks = st.sidebar.slider("Select Forecast Weeks", 1, 52, 12)

    # Prepare data for Prophet (aggregating to weekly level and including external data)
    weekly_df = filtered_df[['Date', 'Variant Inventory Qty', 'Competitor Price', 'Economic Indicator']].copy()
    weekly_df = weekly_df.resample('W-Mon', on='Date').mean().reset_index().sort_values('Date')
    weekly_df = weekly_df.rename(columns={'Date': 'ds', 'Variant Inventory Qty': 'y'})

    # Initialize and train the Prophet model with external regressors
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.add_regressor('Competitor Price')
    model.add_regressor('Economic Indicator')
    model.fit(weekly_df)

    # Create future dates dataframe
    future_weeks = model.make_future_dataframe(periods=forecast_weeks, freq='W-Mon')
    future_weeks['Competitor Price'] = weekly_df['Competitor Price'].mean()  # Assume constant competitor price for simplicity
    future_weeks['Economic Indicator'] = weekly_df['Economic Indicator'].mean()  # Assume constant economic indicator for simplicity
    
    # Predict future values
    forecast_weeks = model.predict(future_weeks)

    # Plot the forecast
    forecast_weeks_fig = px.line(forecast_weeks, x='ds', y='yhat', title=f'Weekly Inventory Forecast for next {forecast_weeks} weeks')
    forecast_weeks_fig.add_scatter(x=weekly_df['ds'], y=weekly_df['y'], mode='markers', name='Actual')
    st.plotly_chart(forecast_weeks_fig)

    # Weekly Stock Adjustment Recommendations
    st.subheader("Weekly Stock Adjustment Recommendations (with External Data)")
    
    weekly_adjustments = []
    for _, row in forecast_weeks.iterrows():
        adjustment = row['yhat'] - (row['yhat'] * 0.10)  # Example: Adjusting stock to be 10% above forecast
        weekly_adjustments.append({
            "Week": row['ds'].strftime('%Y-%m-%d'),
            "Forecasted Quantity": row['yhat'],
            "Suggested Adjustment": adjustment
        })

    # Convert suggestions to DataFrame for better visualization
    if weekly_adjustments:
        adjustments_df = pd.DataFrame(weekly_adjustments)
        st.write("Stock Adjustment Suggestions:")
        
        # Apply color scale to Suggested Adjustment
        styled_adjustments_df = adjustments_df.style.format({"Forecasted Quantity": "{:.2f}", "Suggested Adjustment": "{:.2f}"}).background_gradient(subset=['Suggested Adjustment'], cmap='Greens')
        
        # Use st.dataframe to display with color scales
        st.dataframe(styled_adjustments_df)
        
        # Display as a bar chart
        fig = px.bar(adjustments_df, x='Week', y='Suggested Adjustment', color='Forecasted Quantity',
                     title="Weekly Stock Adjustments", labels={'Suggested Adjustment': 'Adjustment'})
        st.plotly_chart(fig)
    else:
        st.write("No stock adjustments required based on current data.")

    # Automated Inventory Optimization and Market Adaptation
# Automated Inventory Optimization and Market Adaptation
st.subheader("Automated Inventory Optimization and Market Adaptation")

# Example Optimization: Suggest Reorder Quantity based on forecast and market adaptation
reorder_point = 50  # Example threshold for reorder point
safety_stock = 20    # Example safety stock level

optimization_suggestions = []
for _, row in filtered_df.iterrows():
    # Ensure the date exists in the forecast
    matching_forecast = forecast_weeks.loc[forecast_weeks['ds'] == row['Date']]
    
    if not matching_forecast.empty:
        forecasted_qty = matching_forecast['yhat'].values[0]
        if forecasted_qty < reorder_point:
            reorder_qty = reorder_point + safety_stock - row['Variant Inventory Qty']
            optimization_suggestions.append({
                "Product": row['Title'],
                "Vendor": row['Vendor'],
                "Reorder Quantity": reorder_qty
            })
    else:
        st.warning(f"No matching forecast found for the date {row['Date']} in the weekly forecast data.")

# Convert suggestions to DataFrame for better visualization
if optimization_suggestions:
    suggestions_df = pd.DataFrame(optimization_suggestions)
    st.write("Optimization Suggestions:")
    
    # Apply color scale to Reorder Quantity
    styled_df = suggestions_df.style.format({"Reorder Quantity": "{:.2f}"}).background_gradient(subset=['Reorder Quantity'], cmap='Blues')
    
    # Use st.dataframe to display with conditional formatting
    st.dataframe(styled_df)
    
    # Display as a bar chart
    fig = px.bar(suggestions_df, x='Product', y='Reorder Quantity', color='Vendor',
                 title="Reorder Quantity by Product", labels={'Reorder Quantity': 'Reorder Qty'})
    st.plotly_chart(fig)
else:
    st.write("No optimization actions required based on current data.")
st.subheader("Weekly Stock Adjustment Recommendations")
    
weekly_adjustments = []
for _, row in forecast_weeks.iterrows():
            adjustment = row['yhat'] - (row['yhat'] * 0.10)  # Example: Adjusting stock to be 10% above forecast
            weekly_adjustments.append({
                "Week": row['ds'].strftime('%Y-%m-%d'),
                "Forecasted Quantity": row['yhat'],
                "Suggested Adjustment": adjustment
            })

        # Convert suggestions to DataFrame for better visualization
if weekly_adjustments:
            adjustments_df = pd.DataFrame(weekly_adjustments)
            st.write("Stock Adjustment Suggestions:")
            
            # Apply color scale to Suggested Adjustment
            styled_adjustments_df = adjustments_df.style.format({"Forecasted Quantity": "{:.2f}", "Suggested Adjustment": "{:.2f}"}).background_gradient(subset=['Suggested Adjustment'], cmap='Greens')
            
            # Use st.dataframe to display with color scales
            st.dataframe(styled_adjustments_df)
            
            # Display as a bar chart
            fig = px.bar(adjustments_df, x='Week', y='Suggested Adjustment', color='Forecasted Quantity',
                        title="Weekly Stock Adjustments", labels={'Suggested Adjustment': 'Adjustment'})
            st.plotly_chart(fig)
else:
            st.write("No stock adjustments required based on current data.")
   
# Download filtered data as CSV
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_inventory_data.csv',
    mime='text/csv'
)
