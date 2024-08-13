import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px

# Load your data
data_path = '../mnt/data/AmazonSaleReport.csv'  # Replace with your actual CSV file path
df = pd.read_csv(data_path, encoding='ISO-8859-1')

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Handle missing values
df['Amount'] = df['Amount'].fillna(0)

# Sidebar for filtering
st.sidebar.header('Filter Options')
category = st.sidebar.selectbox('Select Category', df['Category'].unique())

# Filtered data based on selection
filtered_data = df[df['Category'] == category]

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

# Category Trends Over Time
st.subheader("Category Trends Over Time")
category_trends = df.groupby(['Date', 'Category'])['Amount'].sum().reset_index()
fig_category_trends = px.line(category_trends, x='Date', y='Amount', color='Category', title='Category Trends Over Time')
st.plotly_chart(fig_category_trends)

# Heatmap - CY vs PY Differences across Fulfilment and Sales Channel
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

# Current Market Trend (Sales Over Time)
st.subheader("Current Market Trend: Sales Over Time")
sales_over_time = filtered_data.groupby('Date')['Amount'].sum().reset_index()
fig_sales_over_time = px.line(sales_over_time, x='Date', y='Amount', title='Sales Over Time')
st.plotly_chart(fig_sales_over_time)

# Category Sales Distribution
st.subheader("Category Sales Distribution")
category_distribution = df.groupby('Category')['Amount'].sum().reset_index()
fig_category_distribution = px.pie(category_distribution, values='Amount', names='Category', title='Category Sales Distribution')
st.plotly_chart(fig_category_distribution)

# Top Products by Sales
st.subheader("Top Products by Sales")
top_products = filtered_data.groupby('Style')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).head(10)
fig_top_products = px.bar(top_products, x='Style', y='Amount', title='Top 10 Products by Sales')
st.plotly_chart(fig_top_products)

# Sales by Region
st.subheader("Sales by Region")
sales_by_region = filtered_data.groupby('ship-state')['Amount'].sum().reset_index()
fig_sales_by_region = px.bar(sales_by_region, x='ship-state', y='Amount', title='Sales by Region')
st.plotly_chart(fig_sales_by_region)

# Identify the product with the most and least sales
st.subheader("Product Sales Analysis")
product_sales = filtered_data.groupby('Style')['Amount'].sum().reset_index()
max_sales_product = product_sales.loc[product_sales['Amount'].idxmax()]
min_sales_product = product_sales.loc[product_sales['Amount'].idxmin()]

st.write(f"**Product with the Most Sales:** {max_sales_product['Style']} - ${max_sales_product['Amount']:.2f}")
st.write(f"**Product with the Least Sales:** {min_sales_product['Style']} - ${min_sales_product['Amount']:.2f}")
