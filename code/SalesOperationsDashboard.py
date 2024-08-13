import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor

class SalesOperationsDashboard:
    def __init__(self, data_path):
        self.data_path = data_path
        self.encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']

    def read_data(self):
        for encoding in self.encodings:
            try:
                data = pd.read_csv(self.data_path, encoding=encoding, on_bad_lines='skip', low_memory=False)
                print(f"Successfully read the file with encoding: {encoding}")
                print(data.columns)
                return data
            except UnicodeDecodeError as e:
                print(f"Error reading CSV file with encoding {encoding}: {e}")
            except Exception as e:
                print(f"Error reading CSV file: {e}")
        return None

    def get_forecast(self, periods=365):
        data = self.read_data()
        if data is None:
            return None, None

        data.ffill(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.dropna(subset=['Date'], inplace=True)
        data.set_index('Date', inplace=True)

        df = data[['Amount', 'Category']].reset_index().rename(columns={'Date': 'ds', 'Amount': 'y'})
        df = df.groupby(['ds', 'Category']).sum().reset_index()

        model = Prophet()
        model.fit(df[['ds', 'y']])

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return df, forecast

    def create_forecast_figure(self, selected_category='all', adjustment_percent=None, periods=365):
        df, forecast = self.get_forecast(periods)
        if df is None or forecast is None:
            return None

        if selected_category != 'all':
            df = df[df['Category'] == selected_category]
            forecast = forecast[forecast['ds'].isin(df['ds'])]

        if adjustment_percent is not None:
            forecast['yhat'] *= (1 + adjustment_percent / 100)

        fig = make_subplots(rows=1, cols=1)

        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Sales'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

        fig.update_layout(title=f'Sales Forecast for {selected_category}', xaxis_title='Date', yaxis_title='Sales')

        return fig

    def run_uplift_prediction(self):
        data = self.read_data()
        if data is None:
            return 'Error loading data.'

        features = data[['Qty', 'Amount']].fillna(0)
        target = data['Amount'].fillna(0) * 1.1
        
        model = RandomForestRegressor()
        model.fit(features, target)

        uplift_predictions = model.predict(features)
        data['Uplift'] = uplift_predictions

        return data

    def create_uplift_figure(self):
        data = self.run_uplift_prediction()
        if isinstance(data, str):
            return None

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=data['Category'],
            y=data['Uplift'],
            name='Uplift',
            marker_color='indianred'
        ))

        fig.update_layout(title='Uplift Prediction by Category', xaxis_title='Category', yaxis_title='Uplift Amount')

        return fig

    def create_sales_breakdown_figure(self):
        data = self.read_data()
        if data is None:
            return None

        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.dropna(subset=['Date'], inplace=True)
        data.set_index('Date', inplace=True)

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'bar'}]])

        fig.add_trace(go.Pie(labels=data['Category'], values=data['Amount'], name='Sales Breakdown'), row=1, col=1)

        data['Month'] = data.index.to_period('M')
        monthly_sales = data.groupby('Month')['Amount'].sum().reset_index()
        fig.add_trace(go.Bar(x=monthly_sales['Month'].astype(str), y=monthly_sales['Amount'], name='Monthly Sales'), row=1, col=2)

        fig.update_layout(title='Sales Breakdown', showlegend=False)

        return fig

    def create_trend_analysis_figure(self, selected_category, selected_retailer):
        data = self.read_data()
        if data is None:
            return None
        
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.dropna(subset=['Date'], inplace=True)
        data.set_index('Date', inplace=True)

        if selected_category != 'all':
            data = data[data['Category'] == selected_category]
        if selected_retailer != 'all':
            data = data[data['Retailer'] == selected_retailer]

        trend_fig = go.Figure()

        trend_fig.add_trace(go.Scatter(x=data.index, y=data['Amount'], mode='lines', name='Sales Amount'))
        
        trend_fig.update_layout(title='Trend Analysis', xaxis_title='Date', yaxis_title='Amount')
        
        return trend_fig

    def create_category_sales_distribution_figure(self, selected_category):
        data = self.read_data()
        if data is None:
            return None

        fig = go.Figure()

        fig.add_trace(go.Pie(labels=data['Category'], values=data['Amount'], name='Category Sales Distribution'))

        fig.update_layout(title='Category Sales Distribution')

        return fig

    def create_sales_over_time_figure(self, selected_category, selected_retailer):
        data = self.read_data()
        if data is None:
            return None

        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data.dropna(subset=['Date'], inplace=True)
        data.set_index('Date', inplace=True)

        if selected_category != 'all':
            data = data[data['Category'] == selected_category]
        if selected_retailer != 'all':
            data = data[data['Retailer'] == selected_retailer]

        sales_over_time_fig = go.Figure()

        sales_over_time_fig.add_trace(go.Scatter(x=data.index, y=data['Amount'], mode='lines', name='Sales Over Time'))
        
        sales_over_time_fig.update_layout(title='Sales Over Time', xaxis_title='Date', yaxis_title='Amount')

        return sales_over_time_fig

    def create_top_products_by_sales_figure(self, selected_category, selected_retailer):
        data = self.read_data()
        if data is None:
            return None

        if selected_category != 'all':
            data = data[data['Category'] == selected_category]
        if selected_retailer != 'all':
            data = data[data['Retailer'] == selected_retailer]

        top_products = data.groupby('SKU')['Amount'].sum().sort_values(ascending=False).head(10).reset_index()

        top_products_fig = go.Figure()

        top_products_fig.add_trace(go.Bar(
            x=top_products['SKU'],
            y=top_products['Amount'],
            name='Top Products by Sales'
        ))

        top_products_fig.update_layout(title='Top Products by Sales', xaxis_title='Product SKU', yaxis_title='Sales Amount')

        return top_products_fig

    def create_sales_by_region_figure(self, selected_category, selected_retailer):
        data = self.read_data()
        if data is None:
            return None

        if selected_category != 'all':
            data = data[data['Category'] == selected_category]
        if selected_retailer != 'all':
            data = data[data['Retailer'] == selected_retailer]

        sales_by_region = data.groupby('ship-state')['Amount'].sum().reset_index()

        sales_by_region_fig = go.Figure()

        sales_by_region_fig.add_trace(go.Bar(
            x=sales_by_region['ship-state'],
            y=sales_by_region['Amount'],
            name='Sales by Region'
        ))

        sales_by_region_fig.update_layout(title='Sales by Region', xaxis_title='Region', yaxis_title='Sales Amount')

        return sales_by_region_fig
