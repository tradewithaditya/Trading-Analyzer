import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import periodogram
from sklearn.cluster import KMeans

# Streamlit app title
st.title("Trading Analysis Dashboard")

# List of stock symbols (extracted from your notebook)
stock_list = [
    "ABB", "ACC", "APLAPOLLO", "AUBANK", "AARTIIND", "ABBOTINDIA", "ADANIENSOL",
    "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ATGL", "ABCAPITAL", "ABFRL",
    "ALKEM", "AMBUJACEM", "ANGELONE", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "ATUL", "AUROPHARMA", "DMART", "AXISBANK",
    # Add more symbols as needed from your notebook’s data fetching cell
    "JSWENERGY", "TATAPOWER", "RELIANCE", "TCS", "HDFCBANK"
]

# Cache data fetching to improve performance
@st.cache_data
def get_stock_data(symbol):
    try:
        stock = yf.download(symbol + ".NS", period="1y")  # .NS for NSE stocks
        if stock.empty:
            st.error(f"No data found for {symbol}")
            return None
        return stock
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Calculate daily returns
def calculate_returns(df):
    df['Return'] = df['Close'].pct_change().dropna()
    return df

# Plot ACF and PACF
def plot_acf_pacf(df, lags=40):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(df['Return'].dropna(), lags=lags, ax=ax1)
    ax1.set_title("Autocorrelation (ACF)")
    plot_pacf(df['Return'].dropna(), lags=lags, method='ywm', ax=ax2)
    ax2.set_title("Partial Autocorrelation (PACF)")
    plt.tight_layout()
    st.pyplot(fig)

# Spectral analysis
def spectral_analysis(df):
    returns = df['Return'].dropna()
    freq, power = periodogram(returns, fs=1.0)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(freq, power)
    plt.title("Periodogram (Spectral Density)")
    plt.xlabel("Frequency (cycles/day)")
    plt.ylabel("Power")
    st.pyplot(fig)
    peak_freq = freq[np.argmax(power)]
    if peak_freq > 0:
        period = 1 / peak_freq
        st.write(f"Dominant cycle: ~{period:.2f} days")
    else:
        st.write("No clear dominant cycle detected.")

# Cluster patterns
def cluster_patterns(df, window_size=10, n_clusters=3):
    returns = df['Close'].pct_change().dropna()
    if len(returns) < window_size:
        st.error("Not enough data for clustering.")
        return None
    X = np.array([returns[i:i+window_size].values for i in range(len(returns) - window_size)])
    if len(X) < n_clusters:
        st.error("Not enough windows for clustering.")
        return None
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    fig = plt.figure(figsize=(12, 6))
    plot_dates = returns.index[window_size - 1:window_size - 1 + len(labels)]
    plt.plot(plot_dates, labels, label="Cluster Label")
    plt.title("Pattern Clusters Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cluster Label")
    plt.legend()
    st.pyplot(fig)
    return labels

# Sidebar for analysis selection
analysis_type = st.sidebar.selectbox("Select Analysis", [
    "ACF/PACF", "Spectral Analysis", "Clustering"
])

# Main app logic
if analysis_type == "ACF/PACF":
    stock = st.selectbox("Select Stock", stock_list)
    if st.button("Analyze"):
        df = get_stock_data(stock)
        if df is not None:
            df = calculate_returns(df)
            plot_acf_pacf(df)

elif analysis_type == "Spectral Analysis":
    stock = st.selectbox("Select Stock", stock_list)
    if st.button("Analyze"):
        df = get_stock_data(stock)
        if df is not None:
            df = calculate_returns(df)
            spectral_analysis(df)

elif analysis_type == "Clustering":
    stock = st.selectbox("Select Stock", stock_list)
    window_size = st.slider("Window Size", 5, 20, 10)
    n_clusters = st.slider("Number of Clusters", 2, 5, 3)
    if st.button("Analyze"):
        with st.spinner("Performing clustering..."):
            df = get_stock_data(stock)
            if df is not None:
                df = calculate_returns(df)
                cluster_patterns(df, window_size, n_clusters)
