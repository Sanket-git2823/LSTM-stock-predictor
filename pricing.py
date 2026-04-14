#run cmd - streamlit run 
import math
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  #MainMenu, footer, header {visibility: hidden;}
  .block-container { padding-top: 1.5rem; }
  .metric-card {
    background: #111418;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label { font-size: 11px; color: #5a6a7a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
  .metric-value { font-size: 26px; font-weight: 700; color: #c8d4e0; }
  .metric-sub   { font-size: 11px; color: #4a5a6a; margin-top: 3px; }
  div[data-testid="stSidebar"] { background: #111418; border-right: 1px solid #1e2530; }
  .stButton button {
    width: 100%; background: #7c6ff7; color: white;
    border: none; font-weight: 600; letter-spacing: 0.08em;
    padding: 12px; border-radius: 6px; font-size: 13px;
  }
  .stButton button:hover { background: #6a5de0; border: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")
    df = df[["Close"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def prepare_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm(lookback: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_and_predict(ticker, period, epochs, forecast_days, lookback=60):
    df      = fetch_stock_data(ticker, period)
    prices  = df["Close"].values.reshape(-1, 1)
    dates   = df.index

    scaler  = MinMaxScaler(feature_range=(0, 1))
    scaled  = scaler.fit_transform(prices)

    split   = int(len(scaled) * 0.8)
    train   = scaled[:split]
    test    = scaled[split - lookback:]

    X_train, y_train = prepare_sequences(train, lookback)
    X_test,  y_test  = prepare_sequences(test,  lookback)

    X_train = X_train.reshape(-1, lookback, 1)
    X_test  = X_test.reshape(-1,  lookback, 1)

    model = build_lstm(lookback)

    progress = st.progress(0, text="Training LSTM...")
    for epoch in range(epochs):
        model.fit(X_train, y_train, epochs=1, batch_size=32,
                  validation_split=0.1, verbose=0)
        pct = int((epoch + 1) / epochs * 100)
        progress.progress(pct, text=f"Training LSTM — epoch {epoch+1}/{epochs}")
    progress.empty()

    pred_scaled = model.predict(X_test, verbose=0)
    pred_prices = scaler.inverse_transform(pred_scaled).flatten()
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_dates  = dates[split:]

    rmse = math.sqrt(mean_squared_error(real_prices, pred_prices))
    mae  = mean_absolute_error(real_prices, pred_prices)
    mape = float(np.mean(np.abs((real_prices - pred_prices) / real_prices)) * 100)

    # Future forecast
    last_window  = scaled[-lookback:].reshape(1, lookback, 1)
    future_preds = []
    window = last_window.copy()
    for _ in range(forecast_days):
        nxt = model.predict(window, verbose=0)[0, 0]
        future_preds.append(nxt)
        window = np.roll(window, -1, axis=1)
        window[0, -1, 0] = nxt

    future_prices = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()
    future_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1),
        periods=forecast_days, freq="B"
    )

    return {
        "df": df, "dates": dates, "prices": prices.flatten(),
        "test_dates": test_dates, "real_prices": real_prices, "pred_prices": pred_prices,
        "future_dates": future_dates, "future_prices": future_prices,
        "rmse": rmse, "mae": mae, "mape": mape,
        "last_price": float(prices[-1][0]),
    }


with st.sidebar:
    st.markdown("### 📈 LSTM Stock Predictor")
    st.markdown("---")

    ticker = st.text_input("Stock Ticker", value="AAPL",
                           placeholder="AAPL, TSLA, RELIANCE.NS").upper().strip()

    period = st.selectbox("Data Period",
                          ["1y", "2y", "3y", "5y"], index=1,
                          format_func=lambda x: {"1y":"1 Year","2y":"2 Years","3y":"3 Years","5y":"5 Years"}[x])

    epochs = st.selectbox("Training Epochs",
                          [10, 20, 50], index=1,
                          format_func=lambda x: f"{x} epochs ({'fast ~20s' if x==10 else 'balanced ~40s' if x==20 else 'accurate ~90s'})")

    forecast_days = st.selectbox("Forecast Days", [7, 14, 30, 60], index=2)

    run = st.button("▶  Train & Predict")

    st.markdown("---")
    st.markdown("""
**Indian stocks:** add `.NS`  
e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`

**US stocks:**  
`AAPL`, `TSLA`, `GOOGL`, `MSFT`

**Crypto:**  
`BTC-USD`, `ETH-USD`
    """)


st.title("📈 LSTM Stock Price Predictor")
st.caption("Deep learning · Real market data · Future forecast")

if not run:
    st.info("👈 Configure your stock in the sidebar and click **Train & Predict**")
    st.markdown("### How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Fetch Data**\n\nPulls real OHLCV data from Yahoo Finance")
    with col2:
        st.markdown("**2. Preprocess**\n\nScales data, creates 60-day lookback sequences")
    with col3:
        st.markdown("**3. Train LSTM**\n\nStacked LSTM with dropout regularisation")
    with col4:
        st.markdown("**4. Forecast**\n\nPredicts future prices autoregressively")

else:
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        try:
            with st.spinner(f"Fetching data and training LSTM for **{ticker}**..."):
                result = train_and_predict(ticker, period, epochs, forecast_days)

            r = result

            # ── Metrics ──
            st.markdown("### Performance Metrics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last Price",  f"${r['last_price']:,.2f}", ticker)
            c2.metric("RMSE",        f"${r['rmse']:.2f}",       "root mean sq error")
            c3.metric("MAE",         f"${r['mae']:.2f}",        "mean abs error")
            c4.metric("MAPE",        f"{r['mape']:.2f}%",       "mean abs % error")

            st.markdown("---")

            # ── Chart 1: Actual vs Predicted ──
            st.markdown("### Actual vs Predicted (Test Set)")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=r["test_dates"], y=r["real_prices"],
                name="Actual", line=dict(color="#00d4aa", width=1.5)
            ))
            fig1.add_trace(go.Scatter(
                x=r["test_dates"], y=r["pred_prices"],
                name="LSTM Predicted", line=dict(color="#f7c06f", width=1.5, dash="dot")
            ))
            fig1.update_layout(
                paper_bgcolor="#111418", plot_bgcolor="#0a0c0f",
                font=dict(color="#c8d4e0", family="IBM Plex Mono", size=11),
                legend=dict(bgcolor="#111418", bordercolor="#1e2530", borderwidth=1),
                xaxis=dict(gridcolor="#1e2530", showgrid=True),
                yaxis=dict(gridcolor="#1e2530", showgrid=True, title="Price ($)"),
                margin=dict(l=10, r=10, t=10, b=10), height=320,
            )
            st.plotly_chart(fig1, use_container_width=True)

            # ── Chart 2: Forecast ──
            st.markdown("### Future Price Forecast")
            hist_tail  = r["df"]["Close"].iloc[-120:]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=hist_tail.index, y=hist_tail.values,
                name="Historical", line=dict(color="#00d4aa", width=1.5)
            ))
            # connector line
            fig2.add_trace(go.Scatter(
                x=[hist_tail.index[-1], r["future_dates"][0]],
                y=[float(hist_tail.values[-1]), float(r["future_prices"][0])],
                line=dict(color="#ff6b9d", width=1.5, dash="dot"),
                showlegend=False
            ))
            fig2.add_trace(go.Scatter(
                x=r["future_dates"], y=r["future_prices"],
                name="Forecast", line=dict(color="#ff6b9d", width=2, dash="dot"),
                fill="tozeroy", fillcolor="rgba(255,107,157,0.05)"
            ))
            fig2.update_layout(
                paper_bgcolor="#111418", plot_bgcolor="#0a0c0f",
                font=dict(color="#c8d4e0", family="IBM Plex Mono", size=11),
                legend=dict(bgcolor="#111418", bordercolor="#1e2530", borderwidth=1),
                xaxis=dict(gridcolor="#1e2530", showgrid=True),
                yaxis=dict(gridcolor="#1e2530", showgrid=True, title="Price ($)"),
                margin=dict(l=10, r=10, t=10, b=10), height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # ── Forecast table ──
            st.markdown("### Forecast Table")
            forecast_df = pd.DataFrame({
                "Date":          r["future_dates"].strftime("%Y-%m-%d"),
                "Forecast Price": [f"${p:,.2f}" for p in r["future_prices"]],
                "Change":        [f"{((r['future_prices'][i]-r['last_price'])/r['last_price']*100):+.2f}%"
                                  for i in range(len(r["future_prices"]))]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check your ticker symbol. For Indian stocks use `RELIANCE.NS`, for crypto use `BTC-USD`")