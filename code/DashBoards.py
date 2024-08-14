import streamlit as st
from SalesDashBoard import sales_dashboard
from InventoryDashBoard import inventory_dashboard

# Sidebar for navigation
st.sidebar.header("Dashboard Navigation")
option = st.sidebar.radio("Go to", ('Sales Dashboard', 'Inventory Optimization Dashboard'))

# Display selected dashboard
if option == 'Sales Dashboard':
    sales_dashboard()
elif option == 'Inventory Optimization Dashboard':
    inventory_dashboard()
