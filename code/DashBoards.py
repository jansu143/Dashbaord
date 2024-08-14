import streamlit as st
import pandas as pd
import plotly.express as px

# Import the specific dashboard scripts
from InventoryDashBoard import inventory_dashboard
from SalesDashBoard import sales_dashboard

# Unified sidebar for navigation
st.sidebar.title("Dashboard Navigation")
dashboard_selection = st.sidebar.radio("Go to", ("Sales Dashboard", "Inventory Optimization Dashboard"))

# Conditional rendering based on selection
if dashboard_selection == "Sales Dashboard":
    st.title("AI Trend Analysis (Sales Forecasting with Prophet and Holidays)")
    sales_dashboard()
elif dashboard_selection == "Inventory Optimization Dashboard":
    st.title("Inventory Optimization Dashboard Based on Demand")
    inventory_dashboard()
