#  Bitcoin Price Forecasting Portal

An interactive Streamlit web application designed to analyze historical Bitcoin price data and project future trends using advanced time-series modeling. This tool allows users to upload custom Kaggle datasets, configure model parameters, and visualize technical indicators alongside AI-driven forecasts.


---

## 🚀 Setup Instructions

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Installation
Clone this repository or save the script, then install the dependencies from the provided file:


pip install -r requirements.txt

### 3. Running the App Locally

streamlit run your_filename.py 


## 🚨 CRITICAL OPERATING NOTE

> **[!IMPORTANT] Manual Re-training Required**
>
> If you adjust any of the following:
>
> * **Model Selection** (Prophet vs. ARIMA)
> * **Forecast Horizon** (Number of days)
> * **Confidence Interval**
> * **Technicals** (SMA/EMA Overlays)
>
> You **MUST** click the **🚀 Generate Forecast** button again. This action is required to trigger the backend engine to re-train the model on your data and render the updated projections.

---

---

## 🛠️ Project Architecture & Dependencies
* **Frontend:** Streamlit
* **Forecasting Engines:**
    * **Prophet:** Optimized for growth trends and seasonal shifts.
    * **SARIMAX:** Enhanced with Fourier terms for complex cyclic analysis.
* **Visuals:** Plotly Graph Objects for interactive financial charting.
* **Data Processing:** Pandas for daily resampling and timestamp normalization.

---

## 🧠 Handling Crypto Volatility
Bitcoin is notoriously volatile. This portal addresses the "noise" of the crypto market through specific engineering choices:

* **Resampling & Noise Reduction:** Resamples data to a daily ('D') frequency to stabilize the trend line while retaining significant price movements.
* **Seasonality via Fourier Series (ARIMA):** Captures multi-period cycles (weekly, monthly, yearly) using exogenous Fourier terms:
    $$y_t = \text{Trend} + \sum_{n=1}^{N} [a_n \cos(\frac{2\pi nt}{P}) + b_n \sin(\frac{2\pi nt}{P})]$$
* **Uncertainty Quantification:** Visualizes a Confidence Interval (Uncertainty Zone) to emphasize that projections are probabilistic ranges, not guaranteed targets.

---

## 📂 Datasets Used
This project has been tested and validated using the following datasets from Kaggle:

* **[BTCUSD_1m_Binance](https://www.kaggle.com/datasets/imranbukhari/comprehensive-btcusd-1m-data):** 
* **[Bitcoin Price Dataset (2017-2023)](https://www.kaggle.com/datasets/jkraak/bitcoin-price-dataset):** 


---

---

## 📊 How to Use
1.  **Upload:** Drop a BTC CSV (e.g., from Kaggle).
2.  **Map:** Select which column is the Date and which is the Price.
3.  **Configure:** Choose your algorithm and settings in the sidebar.
4.  **Execute:** **Click "Generate Forecast" to train the model.**
5.  **Analyze:** Review the MAE and RMSE metrics to evaluate model accuracy and observe the output chart.

---