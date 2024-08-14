import streamlit as st

def sales_dashboard():
    st.title("Comprehensive Sales Dashboard")
    
    # Insert all your sales dashboard code here
    st.subheader("AI Trend Analysis (Sales Forecasting with Prophet and Holidays)")
    # (Your sales dashboard code)
    # ...

def inventory_dashboard():
    st.title("Inventory Optimization Dashboard")
    
    # Insert all your inventory dashboard code here
    st.subheader("Inventory Levels Over Time")
    # (Your inventory optimization dashboard code)
    # ...

# Sidebar for navigation
st.sidebar.title("Dashboard Navigation")
dashboard_selection = st.sidebar.radio("Go to", ("Sales Dashboard", "Inventory Optimization Dashboard"))

# Render the selected dashboard
if dashboard_selection == "Sales Dashboard":
    sales_dashboard()
elif dashboard_selection == "Inventory Optimization Dashboard":
    inventory_dashboard()
