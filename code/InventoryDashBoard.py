import streamlit as st
import pandas as pd
import plotly.express as px

# Load the data
@st.cache_data
def inventory_dashboard():
    inventory_data_path = '../mnt/data/apparel.csv'  # Adjust the path as necessary
    df = pd.read_csv(inventory_data_path)
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
vendor = st.sidebar.selectbox('Select Vendor', inventory_df['Vendor'].unique())
product_type = st.sidebar.selectbox('Select Product Type', inventory_df['Type'].unique())

# Filtered data based on selection
filtered_inventory = inventory_df[(inventory_df['Vendor'] == vendor) & 
                                  (inventory_df['Type'] == product_type)]

# Debug: Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_inventory)

# Check if Variant SKU is empty
if filtered_inventory['Variant SKU'].isnull().all():
    st.warning("The 'Variant SKU' column is empty for the selected filters.")
else:
    # Inventory Levels Visualization
    st.subheader("Inventory Levels by SKU")
    inventory_trend = filtered_inventory.groupby(['Variant SKU'])['Variant Inventory Qty'].sum().reset_index()
    fig_inventory_trend = px.bar(inventory_trend, x='Variant SKU', y='Variant Inventory Qty', title='Inventory Levels by SKU')
    st.plotly_chart(fig_inventory_trend)

    # Inventory Optimization (Simple Example)
    st.subheader("Inventory Optimization")

    # Example optimization (adjust inventory levels)
    optimization_results = filtered_inventory.copy()
    optimization_results['Optimized Inventory'] = optimization_results['Variant Inventory Qty'] * 1.1  # Example: Increase by 10%

    fig_optimization = px.bar(optimization_results, x='Variant SKU', y='Optimized Inventory', title='Optimized Inventory Levels by SKU')
    st.plotly_chart(fig_optimization)

    # Display Optimization Results
    st.write(optimization_results[['Variant SKU', 'Variant Inventory Qty', 'Optimized Inventory']])
