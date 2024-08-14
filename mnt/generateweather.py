import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simulate data for 30 days
num_days = 30
start_date = datetime(2023, 8, 1)
date_list = [start_date + timedelta(days=x) for x in range(num_days)]

# Simulated weather data
temperature_avg = np.random.uniform(20, 35, size=num_days)
temperature_max = temperature_avg + np.random.uniform(0, 5, size=num_days)
temperature_min = temperature_avg - np.random.uniform(0, 5, size=num_days)
precipitation = np.random.uniform(0, 10, size=num_days)
wind_speed = np.random.uniform(2, 10, size=num_days)
wind_direction = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], size=num_days)
events = np.random.choice(['', 'Rain', 'Snow', 'Fog', 'Heavy Rain'], size=num_days)
season = ['Summer'] * num_days  # Example with a single season
is_holiday = np.random.choice([0, 1], size=num_days, p=[0.9, 0.1])  # 10% chance of holiday
sales = np.random.uniform(1000, 2000, size=num_days)  # Simulated sales data

# Create DataFrame
weather_sales_df = pd.DataFrame({
    'Date': date_list,
    'Data.Temperature.Avg Temp': temperature_avg,
    'Data.Temperature.Max Temp': temperature_max,
    'Data.Temperature.Min Temp': temperature_min,
    'Data.Precipitation': precipitation,
    'Data.Wind.Speed': wind_speed,
    'Data.Wind.Direction': wind_direction,
    'Event': events,
    'Season': season,
    'is_holiday': is_holiday,
    'Sales': sales
})

# Save to CSV
weather_sales_df.to_csv('weather_sales_data.csv', index=False)
