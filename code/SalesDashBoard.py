import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load your data
data_path = '../mnt/data/AmazonSaleReport.csv'  # Replace with your actual CSV file path
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Handle missing values
df['Amount'] = df['Amount'].fillna(0)

# Load weather data (you need to have this data)
weather_data_path = '../mnt/data/weather_sales_data.csv'  # Replace with your weather data CSV path
weather_df = pd.read_csv(weather_data_path)

# Ensure the 'Date' column is in datetime format
weather_df['Date'] = pd.to_datetime(weather_df['Date.Full'])

# Merge sales data with weather data on the 'Date' column
merged_df = pd.merge(df, weather_df, on='Date', how='left')

# Sidebar for filtering
st.sidebar.header('Filter Options')
category = st.sidebar.selectbox('Select Category', merged_df['Category'].unique())

# Filtered data based on selection
filtered_data = merged_df[merged_df['Category'] == category]

# Define Holidays
holidays = pd.DataFrame({
    'holiday': ['New Year', 'Christmas', 'Independence Day'],
    'ds': pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04']),
    'lower_window': 0,
    'upper_window': 1,
})

# AI Trend Analysis using Prophet with Holidays
st.subheader("AI Trend Analysis (Sales Forecasting with Prophet and Holidays)")
if not filtered_data.empty:
    # Prepare data for Prophet
    prophet_df = filtered_data[['Date', 'Amount']].rename(columns={'Date': 'ds', 'Amount': 'y'})
    
    # Initialize and fit Prophet model with holidays
    model = Prophet(holidays=holidays)
    model.fit(prophet_df)
    
    # Create future dates for prediction
    future_dates = model.make_future_dataframe(periods=30)  # Forecasting 30 days into the future
    forecast = model.predict(future_dates)
    
    # Plot forecast
    fig_forecast = go.Figure()

    # Plot actual data
    fig_forecast.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='lines', name='Actual'))

    # Plot forecast data
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Adding confidence intervals
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', fill='tonexty', name='Upper Confidence Interval'))
    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', name='Lower Confidence Interval'))

    fig_forecast.update_layout(title='Sales Forecasting using Prophet with Holiday Effects', xaxis_title='Date', yaxis_title='Sales Amount')
    st.plotly_chart(fig_forecast)

    # Plot the holiday effects
    st.subheader("Holiday Effects on Sales")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)
else:
    st.write("No data available for the selected category.")

# Customer Trend Analysis
st.subheader("Customer Trend Analysis")
if not filtered_data.empty:
    # Plot customer trends over time (e.g., sales by date)
    customer_trend_fig = px.line(filtered_data, x='Date', y='Amount', title='Customer Trend Over Time')
    st.plotly_chart(customer_trend_fig)
else:
    st.write("No data available for the selected category.")

# AI-Powered Sales Prediction Based on Weather
st.subheader("AI-Powered Sales Prediction Based on Weather")
if not filtered_data.empty:
    # Feature selection: include weather features
    features = ['Data.Temperature.Avg Temp', 'Data.Precipitation'] # Add relevant weather features
    filtered_data['Data.Precipitation'] = filtered_data['Date'].map(pd.Timestamp.toordinal)

    # Prepare the dataset
    X = filtered_data[features + ['Data.Precipitation']]
    y = filtered_data['Amount']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    weather_model = RandomForestRegressor()
    weather_model.fit(X_train, y_train)

    # Make predictions
    predictions = weather_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    st.write(f"Mean Absolute Error: {mae:.2f}")

    # Show the predictions vs actuals
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    st.write(comparison_df.head())

    # Simulate different weather conditions
    st.sidebar.subheader("Weather Simulation")
    temp = st.sidebar.slider('Temperature (°C)', min_value=-10, max_value=40, value=20)
    humidity = st.sidebar.slider('Humidity (%)', min_value=0, max_value=100, value=50)
    precipitation = st.sidebar.slider('Precipitation (mm)', min_value=0, max_value=50, value=10)

    # Use the model to predict sales based on simulated weather conditions
    simulated_weather = pd.DataFrame({
        'Temperature': [temp],
        'Humidity': [humidity],
        'Precipitation': [precipitation],
        'Date_ordinal': [merged_df['Date_ordinal'].max() + 1]  # Future date
    })
    simulated_sales = weather_model.predict(simulated_weather)

    st.write(f"Predicted Sales under these conditions: ${simulated_sales[0]:.2f}")
else:
    st.write("No data available for the selected category.")

# Predict Uplift with Scenario Adjustments
st.subheader("Predict Uplift with Scenario Adjustments")
if not filtered_data.empty:
    # User input for scenario adjustments
    uplift_percentage = st.slider("Expected Increase in Sales (%)", min_value=0, max_value=100, value=10)
    
    # Adjust the forecast based on user input
    adjusted_forecast = forecast.copy()
    adjusted_forecast['yhat'] *= (1 + uplift_percentage / 100.0)

    # Plot the adjusted forecast
    fig_adjusted_forecast = go.Figure()

    # Plot the original forecast data
    fig_adjusted_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Original Forecast'))

    # Plot the adjusted forecast data
    fig_adjusted_forecast.add_trace(go.Scatter(x=adjusted_forecast['ds'], y=adjusted_forecast['yhat'], mode='lines', name='Adjusted Forecast'))

    fig_adjusted_forecast.update_layout(title=f'Sales Forecast with {uplift_percentage}% Uplift in {category}', xaxis_title='Date', yaxis_title='Sales Amount')
    st.plotly_chart(fig_adjusted_forecast)
else:
    st.write("No data available for the selected category.")

# Rest of the Dashboard Code (Category Trends, Heatmaps, etc.)
st.subheader("Category Trends Over Time")
category_trends = merged_df.groupby(['Date', 'Category'])['Amount'].sum().reset_index()
fig_category_trends = px.line(category_trends, x='Date', y='Amount', color='Category', title='Category Trends Over Time')
st.plotly_chart(fig_category_trends)

st.subheader("Heatmap: CY vs PY Differences")
summary_df_extended = filtered_data.groupby(
    ['Category', 'Fulfilment', 'Sales Channel ', 'ship-service-level', 'Status']
).agg({
    'Order ID': 'count',
    'Amount': 'sum',
}).reset_index()

summary_df_extended['PY Actual'] = summary_df_extended['Amount'] * 0.95
summary_df_extended['CY vs PY'] = summary_df_extended['Amount'] - summary_df_extended['PY Actual']

heatmap_data = summary_df_extended.pivot_table(values='CY vs PY',
                                               index='Fulfilment',
                                               columns='Sales Channel ',
                                               aggfunc=np.sum)
fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis'))
fig_heatmap.update_layout(title='CY vs PY Differences Heatmap', xaxis_nticks=36)
st.plotly_chart(fig_heatmap)

st.subheader("Current Market Trend: Sales Over Time")
sales_over_time = filtered_data.groupby('Date')['Amount'].sum().reset_index()
fig_sales_over_time = px.line(sales_over_time, x='Date', y='Amount', title='Sales Over Time')
st.plotly_chart(fig_sales_over_time)

st.subheader("Category Sales Distribution")
category_distribution = merged_df.groupby('Category')['Amount'].sum().reset_index()
fig_category_distribution = px.pie(category_distribution, values='Amount', names='Category', title='Category Sales Distribution')
st.subheader("Category Sales Distribution")
category_distribution = merged_df.groupby('Category')['Amount'].sum().reset_index()
fig_category_distribution = px.pie(category_distribution, values='Amount', names='Category', title='Category Sales Distribution')
st.plotly_chart(fig_category_distribution)

st.subheader("Top Products by Sales")
top_products = filtered_data.groupby('Style')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
fig_top_products = px.bar(top_products, x='Style', y='Amount', title='Top 10 Products by Sales')
st.plotly_chart(fig_top_products)

st.subheader("Sales by Region")
sales_by_region = filtered_data.groupby('ship-state')['Amount'].sum().reset_index()
fig_sales_by_region = px.bar(sales_by_region, x='ship-state', y='Amount', title='Sales by Region')
st.plotly_chart(fig_sales_by_region)

st.subheader("Product Sales Analysis")
product_sales = filtered_data.groupby('Style')['Amount'].sum().reset_index()
max_sales_product = product_sales.loc[product_sales['Amount'].idxmax()]
min_sales_product = product_sales.loc[product_sales['Amount'].idxmin()]

st.write(f"**Product with the Most Sales:** {max_sales_product['Style']} - ${max_sales_product['Amount']:.2f}")
st.write(f"**Product with the Least Sales:** {min_sales_product['Style']} - ${min_sales_product['Amount']:.2f}")
