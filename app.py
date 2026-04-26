import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import Fourier

@st.cache_data
def get_raw_data(file):
    """Loads CSV once and keeps it in memory """
    return pd.read_csv(file)

# Page Config
st.set_page_config(page_title="BTC Forecasting Portal", layout="wide")
st.title("₿ Bitcoin Price Forecasting Portal")
st.markdown("Upload a Kaggle BTC dataset to analyze trends and project future prices.")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("1. Configuration")
    uploaded_file = st.file_uploader("Upload BTC (CSV File)", type=['csv'])
    
    st.divider()
    
    with st.form("main_config_form"):

        # --- SECTION A: COLUMNS ---
        st.subheader("2. Column Identification")
        if uploaded_file is not None:
            # We use the cached function here
            df_temp = get_raw_data(uploaded_file)
            all_cols = df_temp.columns.tolist()
            numeric_cols = df_temp.select_dtypes(include=['number']).columns.tolist()
            
            col_date = st.selectbox("Timestamp Column", options=all_cols, index=None, placeholder="Select date column...")
            col_price = st.selectbox("Price Column", options=numeric_cols, index=None, placeholder="Select close/open price...")
        else:
            st.info("Please upload a file to select columns.")
            col_date, col_price = None, None
        st.divider()

        # --- SECTION B: MODEL SETTINGS ---

        st.header("3. Model Settings")
        model_choice = st.selectbox("Select Algorithm", ["ARIMA","Prophet"])
        horizon = st.slider("Forecast Horizon (Days)", 7, 1460, 30)
        conf_interval = st.select_slider("Confidence Interval", options=[0.80, 0.95], value=0.95)
        st.divider()

        # --- SECTION C: TECHNICALS ---
        st.header("4. Technicals")
        show_sma = st.checkbox("Show 50-Day SMA")
        show_ema = st.checkbox("Show 20-Day EMA")
        st.divider()

        generate_btn = st.form_submit_button("🚀 Generate Forecast", use_container_width=True)

# --- HELPER FUNCTIONS ---
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

# --- MAIN LOGIC ---
if uploaded_file is not None:
    if not col_date or not col_price:
        st.success("✅ File uploaded successfully! Now, please identify the **Date** and **Price** columns in the sidebar. \n\nThen adjust your **Model Settings** in the sidebar and click **Generate Forecast**.")
    
    # Logic only proceeds if the form was submitted
    
    if generate_btn:
        if not col_date or not col_price:
            st.error("❌ Error: You must select both the **Date** and **Price** columns in the sidebar.")
            st.stop()

    
            # A. DATA PREPARATION
        df_raw = get_raw_data(uploaded_file)
    
    


    

        # Pre-processing
        df = df_raw[[col_date, col_price]].copy()

        # --- HANDLE UNIX TIMESTAMP ---
        # Check if the date column contains numeric values (like 1325412060.0)
        if pd.api.types.is_numeric_dtype(df[col_date]):
            # Check if it's in seconds or milliseconds
            sample_value = df[col_date].iloc[0]
            
            # If value > 1e10, it's likely milliseconds (since year 1970 in ms is ~1.6e12)
            # If value is between 1e9 and 1e10, it's likely seconds
            if sample_value > 1e10:  # Milliseconds
                df[col_date] = pd.to_datetime(df[col_date], unit='ms')
            else:  # Seconds (including your 1325412060.0)
                df[col_date] = pd.to_datetime(df[col_date], unit='s')
        else:
            # Regular datetime string
            df[col_date] = pd.to_datetime(df[col_date])

        df = df.sort_values(by=col_date).dropna()
        
        df_p = df.rename(columns={col_date: 'ds', col_price: 'y'})
        
        df_p = df_p.set_index('ds').resample('D').last().dropna().reset_index()

    
        with st.spinner(f"Training {model_choice} model..."):
                #  BACKTESTING SPLIT (80/20)
                split_idx = int(len(df_p) * 0.8)
                train_df = df_p.iloc[:split_idx]
                test_df = df_p.iloc[split_idx:]

                forecast_df = pd.DataFrame()
                mae, rmse = 0, 0

                # FORECASTING ENGINE
                if model_choice == "Prophet":
                    m = Prophet(interval_width=conf_interval, daily_seasonality=True,changepoint_prior_scale=0.8)
                    m.fit(train_df)
                    
                    # Predict for metrics (backtest)
                    future_test = m.make_future_dataframe(periods=len(test_df))
                    forecast_backtest = m.predict(future_test).iloc[-len(test_df):]
                    mae, rmse = calculate_metrics(test_df['y'], forecast_backtest['yhat'])
                    
                    # Predict for future (the actual forecast)
                    m_full = Prophet(interval_width=conf_interval).fit(df_p)
                    future_real = m_full.make_future_dataframe(periods=horizon)
                    forecast_df = m_full.predict(future_real)
                

                elif model_choice == "ARIMA":

                    # === Define seasonal periods ===
                
                    period_1 = 7     # weekly seasonality
                    period_2 = 30    # monthly seasonality (approx)
                    period_3 = 90     # quarterly
                    period_4 = 365    # yearly

                    # Fourier terms (controls flexibility)
                    fourier_1 = Fourier(period=period_1, order=3)
                    fourier_2 = Fourier(period=period_2, order=2)
                    fourier_3 = Fourier(period=period_3, order=2)
                    fourier_4 = Fourier(period=period_4, order=3)

                    # === Build exogenous features ===
                    X1 = fourier_1.in_sample(df_p.index)
                    X2 = fourier_2.in_sample(df_p.index)
                    X3 = fourier_3.in_sample(df_p.index)
                    X4 = fourier_4.in_sample(df_p.index)
                    X = pd.concat([X1, X2,X3, X4], axis=1)

                    # Split exogenous like the data
                    X_train = X.iloc[:len(train_df)]
                    X_test = X.iloc[len(train_df):]

                    # =========================
                    #  Train model
                    # =========================
                    auto_model = auto_arima(
                        train_df['y'],
                        exogenous=X_train,
                        seasonal=False,
                        stepwise=True
                    )

                    best_order = auto_model.order
                    model = SARIMAX(train_df['y'],
                                    order=best_order,
                                    exog=X_train,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

                    model_fit = model.fit(disp=False)

                    # =========================
                    #  Backtest
                    # =========================
                    preds = model_fit.forecast(steps=len(test_df), exog=X_test)

                    mae, rmse = calculate_metrics(test_df['y'], preds)

                    # =========================
                    #  Full model training
                    # =========================
                    full_model = SARIMAX(df_p['y'],
                                        order=best_order,
                                        exog=X,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit(disp=False)
                    
                    # === Future Fourier features ===
                    future_index = range(len(df_p), len(df_p) + horizon)

                    X1_future = fourier_1.out_of_sample(steps=horizon, index=future_index)
                    X2_future = fourier_2.out_of_sample(steps=horizon, index=future_index)
                    X3_future = fourier_3.out_of_sample(steps=horizon, index=future_index)
                    X4_future = fourier_4.out_of_sample(steps=horizon, index=future_index)

                    X_future = pd.concat(
                        [X1_future, X2_future, X3_future, X4_future],
                        axis=1 )
                    

                    # =========================
                    #  Forecast
                    # =========================
                    future_preds = full_model.get_forecast(steps=horizon, exog=X_future)

                    # =========================
                    #  Format output 
                    # =========================
                    last_date = df_p['ds'].max()

                    future_dates = [
                        last_date + datetime.timedelta(days=i)
                        for i in range(1, horizon + 1)
                    ]

                    forecast_df = pd.DataFrame({
                        'ds': future_dates,
                        'yhat': future_preds.predicted_mean,
                        'yhat_lower': future_preds.conf_int().iloc[:, 0],
                        'yhat_upper': future_preds.conf_int().iloc[:, 1]
                    })

                

                # --- DISPLAY METRICS ---
                st.subheader("📊 Model Performance (Backtesting)")
                m1, m2 = st.columns(2)
                m1.metric("MAE", f"${mae:,.2f}")
                m2.metric("RMSE", f"${rmse:,.2f}")
            

                # ---  INTERACTIVE VISUALIZATION ---
                st.subheader("📈 Price Prediction Chart")
                fig = go.Figure()

                # Historical Data
                fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], name="Actual Price", line=dict(color="#1f77b4")))

                # Uncertainty Zone
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat_upper'],
                    mode='lines', line=dict(width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat_lower'],
                    mode='lines', line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(0, 204, 150, 0.2)', name="Uncertainty"
                ))

                # Forecast Trend
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat'],
                    name="Projected Trend", line=dict(color="#00CC96", width=3, dash='dash')
                ))

                # Technical Overlays
                if show_sma:
                    df_p['SMA50'] = df_p['y'].rolling(50).mean()
                    fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['SMA50'], name="50 SMA", line=dict(color='orange', width=1)))
                if show_ema:
                    df_p['EMA20'] = df_p['y'].ewm(span=20).mean()
                    fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['EMA20'], name="20 EMA", line=dict(color='red', width=1)))

                # Vertical marker for "Today"
                last_date = df_p['ds'].max()

            #  Draw the vertical line 
                fig.add_vline(x=last_date, line_dash="dot", line_color="white")

                #  Add the text manually as an annotation
                fig.add_annotation(
                    x=last_date,
                    y=df_p['y'].iloc[-1], # Place it at the last known price point
                    text="Forecast Start",
                    showarrow=True,
                    arrowhead=1,
                    ax=-50, # Shifts text 50 pixels to the left
                    ay=-30, # Shifts text 30 pixels up
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    font=dict(color="black")
                )
                fig.update_layout(template="plotly_dark", hovermode="x unified", height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Download Option
                csv_forecast = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                st.download_button("📥 Download Forecast Data", csv_forecast, "btc_forecast.csv", "text/csv")

else:
    st.info("Please upload a CSV file from the sidebar to begin.")