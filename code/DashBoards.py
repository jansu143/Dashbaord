import streamlit as st
from SalesDashBoard import sales_dashboard
from InventoryDashBoard import inventory_dashboard

# Sidebar for navigation
st.sidebar.title("Dashboard Navigation")
dashboard_selection = st.sidebar.radio("Go to", ("Sales Dashboard", "Inventory Optimization Dashboard"))

# Render the selected dashboard
if dashboard_selection == "Sales Dashboard":
    sales_dashboard()
elif dashboard_selection == "Inventory Optimization Dashboard":
    inventory_dashboard()
