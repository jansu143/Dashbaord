import streamlit as st
import pandas as pd
import plotly.express as px

# Load your data
@st.cache
def inventory_dashboard():
    inventory_data_path = '../mnt/data/apparel.csv'  # Adjust the path as necessary
    df = pd.read_csv(inventory_data_path)
    df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D') # Ensure 'Date' is in datetime format
    return df

inventory_df = inventory_dashboard()

# Streamlit App
st.title("Inventory Optimization Dashboard")

# Display some basic information about the dataset
st.subheader("Data Overview")
st.write("Dimensions: ", inventory_df.shape)
st.write("Missing values:", inventory_df.isnull().sum())
st.dataframe(inventory_df.head())

# Sidebar for filtering
st.sidebar.header('Filter Options')
region = st.sidebar.selectbox('Select Region', inventory_df['Region'].unique())
date_range = st.sidebar.slider('Select Date Range', 
                               value=[inventory_df['Date'].min(), 
                                      inventory_df['Date'].max()])

# Filtered data based on selection
filtered_inventory = inventory_df[(inventory_df['Region'] == region) & 
                                  (inventory_df['Date'].between(date_range[0], date_range[1]))]

# Inventory Levels Over Time
st.subheader("Inventory Levels Over Time")
inventory_trend = filtered_inventory.groupby(['Date', 'Product'])['Inventory'].sum().reset_index()
fig_inventory_trend = px.line(inventory_trend, x='Date', y='Inventory', color='Product', title='Inventory Levels Over Time')
st.plotly_chart(fig_inventory_trend)

# Inventory Optimization (Simple Example)
st.subheader("Inventory Optimization")

# Example optimization (replace with actual optimization logic)
optimization_results = filtered_inventory.groupby(['Date', 'Region'])['Inventory'].sum().reset_index()
optimization_results['Optimized_Inventory'] = optimization_results['Inventory'] * 1.1  # Example: Increase by 10%

fig_optimization = px.bar(optimization_results, x='Date', y='Optimized_Inventory', color='Region', title='Optimized Inventory Allocation')
st.plotly_chart(fig_optimization)

# Display Optimization Results
st.write(optimization_results[['Date', 'Region', 'Optimized_Inventory']])
