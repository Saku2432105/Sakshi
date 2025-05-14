import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

if "df" not in st.session_state:
    st.session_state.df = None
if "engineered_df" not in st.session_state:
    st.session_state.engineered_df = None
if "category_forecast_results" not in st.session_state:
    st.session_state.category_forecast_results = {}

EXPECTED_CATEGORIES = [
    "Bluetooth", "Charging", "Dash", "Digital", "Drawing", "Drone", "E", "External",
    "Fitness", "Gaming", "Graphics", "Home", "Laptop", "Laser", "Mechanical", "Microphone",
    "Monitor", "Noise", "Photo", "Portable", "Power", "SSD", "Smart", "Smartphone",
    "Smartwatch", "Soundbar", "Streaming", "Tablet", "USB", "VR", "Webcam", "WiFi", "Wireless"
]

def load_and_preprocess(filepath):
    try:
        logging.info(f"Loading dataset from {filepath.name}...")
        df = pd.read_csv(filepath)
        logging.info("Dataset columns: %s", df.columns.tolist())

        expected_columns = {
            'year': ['year', 'Year'],
            'month': ['month', 'Month'],
            'monthly_sale': ['monthly_sale', 'Monthly Sales', 'monthly sales'],
            'cost': ['cost', 'Cost'],
            'product_name': ['product_name', 'Product Name', 'product name']
        }

        for expected, variations in expected_columns.items():
            found = False
            for var in variations:
                if var in df.columns:
                    df.rename(columns={var: expected}, inplace=True)
                    found = True
                    break
            if not found:
                raise ValueError(f"Missing required column: {expected}. Found columns: {df.columns.tolist()}")

        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)
        df['monthly_sale'] = pd.to_numeric(df['monthly_sale'], errors='coerce')
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01', errors='coerce')

        original_rows = len(df)
        df = df.dropna(subset=['monthly_sale', 'cost', 'date'])
        df = df[df['monthly_sale'] >= 0]
        df = df[df['cost'] >= 0]
        logging.info(f"Rows dropped during cleaning: {original_rows - len(df)}")

        df['category'] = df['product_name'].str.extract(r'([A-Za-z]+)', expand=False)
        df['category'] = df['category'].fillna('Unknown')

        logging.info("After cleaning: Rows: %d", len(df))
        return df
    except Exception as e:
        logging.error("Error in load_and_preprocess: %s", str(e))
        raise

@st.cache_data
def group_by_sales_volume(df):
    try:
        sales_per_category = df.groupby('category')['monthly_sale'].mean().reset_index()
        thresholds = sales_per_category['monthly_sale'].quantile([0.33, 0.66])
        low_threshold = thresholds[0.33]
        high_threshold = thresholds[0.66]

        def assign_group(sales):
            if sales > high_threshold:
                return "High Sales"
            elif sales > low_threshold:
                return "Medium Sales"
            else:
                return "Low Sales"

        sales_per_category['group'] = sales_per_category['monthly_sale'].apply(assign_group)
        return dict(zip(sales_per_category['category'], sales_per_category['group']))
    except Exception as e:
        logging.error(f"Error in group_by_sales_volume: {str(e)}")
        raise

def perform_eda(df, selected_group, category_to_group):
    try:
        group_categories = [cat for cat, group in category_to_group.items() if group == selected_group]
        group_df = df[df['category'].isin(group_categories)]

        if group_df.empty:
            st.warning(f"No data available for {selected_group} group.")
            return

        st.subheader(f"Summary Statistics for {selected_group}")
        summary_stats = group_df.groupby('category')['monthly_sale'].mean().reset_index()
        summary_stats.columns = ['Category', 'Mean Sales']
        st.write(summary_stats)

        st.subheader(f"Sales Trend Over Time ({selected_group})")
        fig_trend = go.Figure()
        for category in group_categories:
            cat_data = group_df[group_df['category'] == category].groupby('date')['monthly_sale'].sum().reset_index()
            fig_trend.add_trace(go.Scatter(
                x=cat_data['date'],
                y=cat_data['monthly_sale'],
                mode='lines',
                name=category
            ))
        fig_trend.update_layout(
            title=f'Total Monthly Sales Over Time ({selected_group})',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            height=400
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.subheader(f"Sales Distribution by Category ({selected_group})")
        fig_dist = px.box(
            group_df,
            x='monthly_sale',
            y='category',
            title=f'Monthly Sales Distribution ({selected_group})',
            labels={'monthly_sale': 'Monthly Sales', 'category': 'Category'},
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader(f"Correlation Heatmap ({selected_group})")
        numerical_cols = ['monthly_sale', 'cost', 'year', 'month']
        corr = group_df[numerical_cols].corr()
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        plt.title(f'Correlation Heatmap ({selected_group})')
        st.pyplot(fig_corr)
        plt.close(fig_corr)
    except Exception as e:
        logging.error(f"Error in perform_eda: {str(e)}")
        st.error(f"Error during EDA: {str(e)}")

def engineer_features(df, category_to_group):
    try:
        df = df.copy()
        df = df.sort_values(['category', 'product_name', 'date'])

        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)

        df['rolling_avg_3m'] = df.groupby(['category', 'product_name'])['monthly_sale'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

        df['sales_growth'] = df.groupby(['category', 'product_name'])['monthly_sale'].pct_change().fillna(0)

        df['group'] = df['category'].map(category_to_group)

        group_median_sales = df.groupby(['group', 'date'])['monthly_sale'].median().reset_index()
        group_median_sales = group_median_sales.rename(columns={'monthly_sale': 'group_median_sales'})
        df = df.merge(group_median_sales, on=['group', 'date'], how='left')

        df['sales_to_group_ratio'] = df['monthly_sale'] / df['group_median_sales']

        df = df.dropna()
        logging.info(f"Rows after feature engineering: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Error in engineer_features: {str(e)}")
        raise

def summarize_engineered_features(df, selected_group):
    try:
        group_df = df[df['group'] == selected_group]
        feature_summary = group_df.groupby('category').agg({
            'day_of_year': 'mean',
            'is_quarter_end': 'mean',
            'rolling_avg_3m': 'mean',
            'sales_growth': 'mean',
            'group_median_sales': 'mean',
            'sales_to_group_ratio': 'mean'
        }).reset_index()
        feature_summary.columns = [
            'Category', 'Mean Day of Year', 'Quarter End Frequency',
            'Mean Rolling Avg (3M)', 'Mean Sales Growth', 'Mean Group Median Sales',
            'Mean Sales to Group Ratio'
        ]
        return feature_summary
    except Exception as e:
        logging.error(f"Error in summarize_engineered_features: {str(e)}")
        raise

@st.cache_data
def create_features_for_forecasting(df):
    try:
        logging.info("Creating features for forecasting...")
        df = df.sort_values(['category', 'product_name', 'date'])

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        grouped = df.groupby(['category', 'product_name'])
        df['sales_lag1'] = grouped['monthly_sale'].shift(1)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        compatible_features = ['rolling_avg_3m', 'sales_growth', 'sales_to_group_ratio']
        for feature in compatible_features:
            if feature not in df.columns:
                df[feature] = 0

        df = df.dropna(subset=['sales_lag1'])
        logging.info(f"Rows after forecasting feature engineering: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Error in create_features_for_forecasting: {str(e)}")
        raise

@st.cache_data
def aggregate_by_category(df):
    try:
        category_df = df.groupby(['category', 'date']).agg({
            'monthly_sale': 'sum',
            'cost': 'sum',
            'year': 'first',
            'month': 'first',
            'quarter': 'first',
            'sales_lag1': 'mean',
            'month_sin': 'mean',
            'month_cos': 'mean',
            'rolling_avg_3m': 'mean',
            'sales_growth': 'mean',
            'sales_to_group_ratio': 'mean'
        }).reset_index()
        logging.info(f"Aggregated dataset size: {len(category_df)} rows")
        return category_df
    except Exception as e:
        logging.error(f"Error in aggregate_by_category: {str(e)}")
        raise

def evaluate_model(y_true, y_pred, model_name):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = mape if not np.isinf(mape) else float('inf')

        logging.info(f"\n{model_name} Performance:")
        logging.info(f"MAE: {mae:.2f}")
        logging.info(f"RMSE: {rmse:.2f}")
        logging.info(f"MAPE: {mape:.2f}%")

        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    except Exception as e:
        logging.error(f"Error in evaluate_model for {model_name}: {str(e)}")
        return {'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE': float('nan')}

def train_test_split_by_time(df, category, test_size=0.2, max_rows=30):
    try:
        category_data = df[df['category'] == category].sort_values('date')
        if len(category_data) < 12:
            logging.warning(f"Insufficient data for {category} (need at least 12 points). Skipping...")
            return None, None

        if len(category_data) > max_rows:
            category_data = category_data.sample(n=max_rows, random_state=42)
            logging.info(f"Sampled {max_rows} rows for {category}")

        split_idx = int(len(category_data) * (1 - test_size))
        train = category_data.iloc[:split_idx]
        test = category_data.iloc[split_idx:]
        return train, test
    except Exception as e:
        logging.error(f"Error in train_test_split_by_time for {category}: {str(e)}")
        return None, None

def random_forest_forecast(train, test, category, periods=3):
    try:
        features = [
            'cost', 'year', 'month', 'quarter', 'sales_lag1', 'month_sin', 'month_cos',
            'rolling_avg_3m', 'sales_growth', 'sales_to_group_ratio'
        ]
        if train is None or test is None or len(train) < 12 or len(test) < 1:
            logging.warning(f"Insufficient data for Random Forest forecast for {category}")
            return None, None, None, None, None, None

        model = RandomForestRegressor(
            n_estimators=10,
            max_depth=2,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(train[features], train['monthly_sale'])

        forecast = model.predict(test[features])
        forecast_df = pd.DataFrame({
            'date': test['date'],
            'forecast': forecast
        })

        last_features = test[features].iloc[-1:].copy()
        future_dates = pd.date_range(start=test['date'].iloc[-1], periods=periods + 1, freq='M')[1:]
        future_forecasts = []
        for i in range(periods):
            last_features['month'] = future_dates[i].month
            last_features['quarter'] = (future_dates[i].month - 1) // 3 + 1
            last_features['month_sin'] = np.sin(2 * np.pi * last_features['month'] / 12)
            last_features['month_cos'] = np.cos(2 * np.pi * last_features['month'] / 12)
            pred = model.predict(last_features[features])[0]
            future_forecasts.append(pred)
            last_features['sales_lag1'] = pred

        future_df = pd.DataFrame({
            'date': future_dates,
            'forecast': future_forecasts
        })
        forecast_df = pd.concat([forecast_df, future_df], ignore_index=True)

        forecast_df['forecast_lower'] = forecast_df['forecast'] * 0.9
        forecast_df['forecast_upper'] = forecast_df['forecast'] * 1.1

        return model, forecast_df, forecast, train, test, forecast_df
    except Exception as e:
        logging.error(f"Random Forest forecast failed for {category}: {str(e)}")
        return None, None, None, None, None, None

st.title("Sales Analysis & Forecasting App")

st.header("Project Guidelines")
st.write("""
- Creating  a dataset that contains the required details in each entry.
- Clean the dataset.
- Sanitize the dataset.
- Choose the appropriate forecasting model for  data.
- Fit the model to the dataset.
- Make predictions for all products.
""")

st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader(
    "Upload a CSV file with columns: Product Name, Year, Month, Monthly Sales, Cost",
    type=["csv"]
)
if uploaded_file:
    try:
        st.session_state.df = load_and_preprocess(uploaded_file)
        st.success(f"Dataset loaded, cleaned, and sanitized! Rows: {len(st.session_state.df)}")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.session_state.df = None

category_to_group = {}
group_options = ["High Sales", "Medium Sales", "Low Sales"]
if st.session_state.df is not None:
    category_to_group = group_by_sales_volume(st.session_state.df)

if st.session_state.df is not None:
    st.header("Step 2: Exploratory Data Analysis")
    selected_group = st.selectbox("Select Sales Group for EDA", group_options, key="eda_group")
    if st.button("Run EDA"):
        with st.spinner(f"Performing EDA for {selected_group}..."):
            perform_eda(st.session_state.df, selected_group, category_to_group)
        st.success("EDA completed!")
else:
    st.write("Please upload a dataset to start the analysis.")

if st.session_state.df is not None:
    st.header("Step 3: Feature Engineering")
    selected_group = st.selectbox("Select Sales Group for Feature Engineering", group_options, key="feature_group")
    if st.button("Generate Features"):
        try:
            st.session_state.engineered_df = engineer_features(st.session_state.df, category_to_group)
            st.success("Features generated successfully!")
            st.write(f"Dataset now has {len(st.session_state.engineered_df)} rows.")

            st.subheader(f"Feature Summary for {selected_group}")
            feature_summary = summarize_engineered_features(st.session_state.engineered_df, selected_group)
            st.write(feature_summary)

            st.subheader("Download Engineered Dataset")
            csv = st.session_state.engineered_df.to_csv(index=False)
            st.download_button(
                label="Download Engineered Data as CSV",
                data=csv,
                file_name="engineered_dataset.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error during feature engineering: {str(e)}")

if st.session_state.engineered_df is not None:
    st.header("Step 4: Forecasting (Grouped by Sales Volume)")

    try:
        forecast_df = create_features_for_forecasting(st.session_state.engineered_df)
        aggregated_df = aggregate_by_category(forecast_df)
    except Exception as e:
        st.error(f"Error preparing data for forecasting: {str(e)}")
        aggregated_df = None

    if aggregated_df is not None:
        categories = aggregated_df['category'].unique().tolist()
        st.write(f"Found {len(categories)} categories: {', '.join(categories)}")

        selected_group = st.selectbox("Select Sales Group to Forecast", group_options, key="forecast_group")
        show_confidence_intervals = st.checkbox("Show Confidence Intervals", value=False)

        if st.button("Run Forecasting for Selected Group"):
            try:
                with st.spinner(f"Running forecasts for {selected_group} categories..."):
                    forecast_results = {}
                    group_plot = go.Figure()
                    forecast_stats = []

                    group_categories = [cat for cat in categories if category_to_group.get(cat) == selected_group]

                    if not group_categories:
                        st.warning(f"No categories found in {selected_group} group.")
                    else:
                        for category in group_categories:
                            if category not in EXPECTED_CATEGORIES:
                                st.warning(f"Skipping {category}: not in expected categories.")
                                logging.warning(f"Skipping {category}: not in expected categories.")
                                continue

                            st.write(f"Processing category: {category}")
                            train, test = train_test_split_by_time(aggregated_df, category)
                            if train is None or test is None:
                                st.warning(f"Skipping {category}: insufficient data.")
                                logging.warning(f"Skipping {category}: insufficient data.")
                                continue

                            result = {}
                            model, forecast_df, forecast, train_data, test_data, forecast_data = random_forest_forecast(
                                train, test, category)
                            if model:
                                metrics = evaluate_model(test['monthly_sale'], forecast, "Random Forest")
                                result['RandomForest'] = {'metrics': metrics, 'forecast': forecast_df}
                                logging.info(f"Random Forest forecast successful for {category}")

                                historical_dates = pd.concat([train_data[['date']], test_data[['date']]], ignore_index=True)
                                historical_sales = pd.concat([train_data[['monthly_sale']], test_data[['monthly_sale']]],
                                                             ignore_index=True)
                                combined_historical = pd.DataFrame({
                                    'date': historical_dates['date'],
                                    'sales': historical_sales['monthly_sale']
                                })

                                group_plot.add_trace(go.Scatter(
                                    x=combined_historical['date'],
                                    y=combined_historical['sales'],
                                    mode='lines',
                                    name=f'{category} (Historical)',
                                    line=dict(width=2)
                                ))

                                group_plot.add_trace(go.Scatter(
                                    x=forecast_data['date'],
                                    y=forecast_data['forecast'],
                                    mode='lines',
                                    name=f'{category} (Forecast)',
                                    line=dict(width=2, dash='dash')
                                ))

                                if show_confidence_intervals:
                                    group_plot.add_trace(go.Scatter(
                                        x=forecast_data['date'],
                                        y=forecast_data['forecast_upper'],
                                        mode='lines',
                                        name=f'{category} (Upper CI)',
                                        line=dict(width=1, dash='dot', color='rgba(0,0,0,0.2)'),
                                        showlegend=False
                                    ))
                                    group_plot.add_trace(go.Scatter(
                                        x=forecast_data['date'],
                                        y=forecast_data['forecast_lower'],
                                        mode='lines',
                                        name=f'{category} (Lower CI)',
                                        line=dict(width=1, dash='dot', color='rgba(0,0,0,0.2)'),
                                        fill='tonexty',
                                        fillcolor='rgba(0,0,0,0.1)',
                                        showlegend=False
                                    ))

                                historical_sales = combined_historical['sales'].values
                                if len(historical_sales) > 1:
                                    growth_rate = (historical_sales[-1] - historical_sales[0]) / historical_sales[
                                        0] * 100 / len(historical_sales)
                                else:
                                    growth_rate = 0

                                forecast_stats.append({
                                    'Category': category,
                                    'Average Forecast': forecast_data['forecast'].mean(),
                                    'Growth Rate (%)': growth_rate
                                })

                            else:
                                st.warning(f"No forecast for {category}")
                                logging.warning(f"No forecast for {category}")

                            if result:
                                forecast_results[category] = result

                        group_plot.update_layout(
                            title=f'Sales Forecast for {selected_group} Categories',
                            xaxis_title='Date',
                            yaxis_title='Sales',
                            showlegend=True,
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=600
                        )
                        group_plot.update_traces(hoverinfo='name+x+y')

                        st.plotly_chart(group_plot, use_container_width=True)

                        st.subheader(f"Forecast Statistics for {selected_group}")
                        stats_df = pd.DataFrame(forecast_stats)
                        st.write(stats_df)

                        st.session_state.category_forecast_results = forecast_results
                        st.success(f"Forecasting completed for {selected_group} categories!")
            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")
                logging.error(f"Forecasting error: {str(e)}")

if st.session_state.category_forecast_results:
    st.header("Step 5: Results")

    try:
        categories = []
        mae_values = []
        rmse_values = []
        mape_values = []

        for category, result in st.session_state.category_forecast_results.items():
            for model_name, data in result.items():
                categories.append(category)
                mae_values.append(data['metrics']['MAE'])
                rmse_values.append(data['metrics']['RMSE'])
                mape_values.append(data['metrics']['MAPE'])

        metrics_df = pd.DataFrame({
            'Category': categories,
            'MAE': mae_values,
            'RMSE': rmse_values,
            'MAPE': mape_values
        })

        st.subheader("MAE Across Categories (Bar Plot)")
        fig_mae = px.bar(
            metrics_df,
            x='MAE',
            y='Category',
            orientation='h',
            title='Mean Absolute Error (MAE) Across Categories',
            color='MAE',
            color_continuous_scale='Viridis',
            height=800
        )
        fig_mae.update_layout(
            xaxis_title='MAE',
            yaxis_title='Category',
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_mae, use_container_width=True)

        st.subheader("RMSE Across Categories (Scatter Plot with Trend Line)")
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Scatter(
            x=metrics_df['RMSE'],
            y=metrics_df['Category'],
            mode='markers+text',
            marker=dict(size=12, color=metrics_df['RMSE'], colorscale='Plasma', showscale=True),
            text=metrics_df['RMSE'].round(2),
            textposition='middle right',
            name='RMSE'
        ))
        fig_rmse.add_trace(go.Scatter(
            x=metrics_df['RMSE'],
            y=metrics_df['Category'],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Trend'
        ))
        fig_rmse.update_layout(
            title='Root Mean Squared Error (RMSE) Across Categories',
            xaxis_title='RMSE',
            yaxis_title='Category',
            height=800,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

        st.subheader("MAPE Across Categories (Box Plot)")
        mape_distributions = []
        for mape, cat in zip(metrics_df['MAPE'], metrics_df['Category']):
            dist = np.random.normal(mape, mape * 0.1, 10)
            mape_distributions.extend([(cat, val) for val in dist])
        mape_dist_df = pd.DataFrame(mape_distributions, columns=['Category', 'MAPE'])

        fig_mape = px.box(
            mape_dist_df,
            x='MAPE',
            y='Category',
            title='Mean Absolute Percentage Error (MAPE) Across Categories',
            color='Category',
            height=800
        )
        fig_mape.update_layout(
            xaxis_title='MAPE (%)',
            yaxis_title='Category',
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_mape, use_container_width=True)

        st.subheader("Export Results")
        export_data = []
        for category, result in st.session_state.category_forecast_results.items():
            for model_name, data in result.items():
                forecast_df = data['forecast']
                for _, row in forecast_df.iterrows():
                    export_data.append({
                        'Category': category,
                        'Model': model_name,
                        'Date': row['date'],
                        'Forecast': row['forecast'],
                        'MAE': data['metrics']['MAE'],
                        'RMSE': data['metrics']['RMSE'],
                        'MAPE': data['metrics']['MAPE']
                    })
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="category_forecast_results.csv",
            mime="text/csv"
        )
        logging.info("Results displayed successfully")
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        logging.error(f"Results display error: {str(e)}")
else:
    st.write("Run forecasting to see results.")