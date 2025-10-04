import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import joblib
import io

# Optional imports for technical indicators & LSTM
try:
    import ta
except Exception:
    ta = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:
    tf = None

"""
This Streamlit application provides a simple interface for downloading historical
stock data, engineering features, training a predictive model, and comparing
the model's predictions to actual future prices. It supports a tabular model
(Random Forest) and, if TensorFlow is installed, an optional LSTM sequence
model.

To run this app:
1. Install the required dependencies listed in `requirements.txt`.
2. Run `streamlit run app.py` from the command line.
3. Interact with the sidebar to select a ticker, date range, and model
   parameters. After fetching data and training the model, the app displays
   charts of actual vs. predicted prices along with evaluation metrics. You
   can download the predictions as a CSV or inspect feature importances for
   the Random Forest model.

This code is provided for educational purposes only and should not be
used as the basis for real investment decisions.
"""

# Configure Streamlit page
st.set_page_config(layout="wide", page_title="Stock Prediction Playground")
st.title("📈 Stock Prediction Playground")
st.write(
    "Download history, train a model, and compare actual vs predicted prices. This is a demo/learning tool, not financial advice."
)

# Sidebar controls
st.sidebar.header("Data & Model Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))
test_size = st.sidebar.slider("Test set size (fraction)", 0.05, 0.5, 0.2, step=0.05)
n_lags = st.sidebar.slider("Number of lag features", 1, 30, 5)
ma_windows = st.sidebar.multiselect(
    "Moving average windows", [5, 10, 20, 50, 100], default=[5, 10, 20]
)
model_choice = st.sidebar.selectbox("Model", ["RandomForest", "LSTM (seq)"])
rf_estimators = st.sidebar.number_input(
    "RF: n_estimators", min_value=10, max_value=1000, value=200, step=10
)
reload_data = st.sidebar.button("Reload data")

# Helper functions
@st.cache_data(ttl=3600)
def fetch_data(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for the given ticker and date range from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g., 'AAPL'.
    start : pd.Timestamp
        Start date for historical data.
    end : pd.Timestamp
        End date for historical data.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns for Open, High, Low, Close, Adj Close, and Volume.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    # Some tickers/data sources might not include an "Adj Close" column. To make the
    # app robust, rename the column if present or create it from "Close" if missing.
    if "Adj Close" in df.columns:
        # Rename to a consistent format for downstream processing
        df = df.rename(columns={"Adj Close": "Adj_Close"})
    else:
        # If there's no adjusted close column, duplicate the close price as Adj_Close
        # so that the rest of the app still works.
        if "Close" in df.columns:
            df["Adj_Close"] = df["Close"]
    # Define the columns we want to keep. Some may be missing depending on the ticker,
    # so we filter by those that actually exist in the DataFrame.
    required_cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
    df = df[[c for c in required_cols if c in df.columns]]
    df.index = pd.to_datetime(df.index)
    return df


def add_features(
    df: pd.DataFrame, n_lags: int = 5, ma_windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Create lagged features, returns, moving averages, and other technical indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the 'Adj_Close' column.
    n_lags : int, optional
        Number of lagged price and return features to compute.
    ma_windows : list of int, optional
        Window sizes for moving average calculations.

    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns and without rows containing NaN values.
    """
    if ma_windows is None:
        ma_windows = [5, 10, 20]
    df = df.copy()
    df["Return"] = df["Adj_Close"].pct_change()
    # lagged prices and returns
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["Adj_Close"].shift(lag)
        df[f"ret_lag_{lag}"] = df["Return"].shift(lag)
    # moving averages and differences
    for w in ma_windows:
        df[f"ma_{w}"] = df["Adj_Close"].rolling(w).mean()
        df[f"ma_diff_{w}"] = df["Adj_Close"] - df[f"ma_{w}"]
    # volatility: rolling standard deviation of returns
    df["vol_10"] = df["Return"].rolling(10).std()
    # RSI (if ta is available)
    if ta is not None:
        try:
            df["rsi_14"] = ta.momentum.RSIIndicator(df["Adj_Close"], window=14).rsi()
        except Exception:
            df["rsi_14"] = np.nan
    else:
        df["rsi_14"] = np.nan
    # Drop rows with NaNs created by shifting/rolling
    df = df.dropna()
    return df


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix for training.
    y_train : np.ndarray
        Target vector for training.
    n_estimators : int
        Number of trees in the forest.

    Returns
    -------
    RandomForestRegressor
        Trained model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def create_sequences(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1-D array of values into input/output sequences for sequential models.

    Parameters
    ----------
    values : np.ndarray
        1-D array of values to sequence.
    seq_len : int
        Length of each input sequence.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing arrays of input sequences and corresponding targets.
    """
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


# Main data fetching
with st.spinner("Fetching data..."):
    data = fetch_data(ticker, start_date, end_date)

if data.empty:
    st.error(
        "No data found for that ticker/date range. Double-check ticker symbol and dates."
    )
    st.stop()

st.write(
    f"Data range: {data.index.min().date()} — {data.index.max().date()}  |  Rows: {len(data)}"
)
st.dataframe(data.tail(5))

# Feature engineering
df = add_features(data, n_lags=n_lags, ma_windows=ma_windows)

# Target: next day Adj_Close (one-step ahead)
df["target"] = df["Adj_Close"].shift(-1)
df = df.dropna()  # removes last row where target is NaN

# Choose features (exclude columns that could leak future information)
exclude_cols = ["target", "Adj_Close", "Close", "Open", "High", "Low"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]
y = df["target"]

st.write(
    f"Using {len(feature_cols)} features: {feature_cols[:8]}{'...' if len(feature_cols) > 8 else ''}"
)

# Train/test split (time series: keep chronological order)
split_idx = int(len(df) * (1 - test_size))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

st.write(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

if model_choice == "RandomForest":
    # Scaling is optional for RandomForest
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = train_random_forest(
        X_train_scaled, y_train, n_estimators=int(rf_estimators)
    )
    preds = model.predict(X_test_scaled)
    # Save model and scaler for potential reuse
    joblib.dump(model, f"rf_{ticker}.joblib")
    joblib.dump(scaler, f"scaler_{ticker}.joblib")
else:
    # LSTM path
    if tf is None:
        st.error(
            "TensorFlow not installed. Install tensorflow in requirements.txt to use LSTM."
        )
        st.stop()
    # Determine sequence length: we use the number of lag features plus the largest MA window as a heuristic
    seq_len = n_lags + (max(ma_windows) if ma_windows else 0)
    vals = df["Adj_Close"].values.reshape(-1, 1)
    mms = MinMaxScaler()
    vals_scaled = mms.fit_transform(vals)
    X_seq, y_seq = create_sequences(vals_scaled, seq_len)
    # Split chronologically
    split_idx_seq = int(len(X_seq) * (1 - test_size))
    X_train_seq, X_test_seq = X_seq[:split_idx_seq], X_seq[split_idx_seq:]
    y_train_seq, y_test_seq = y_seq[:split_idx_seq], y_seq[split_idx_seq:]
    # Reshape for LSTM [samples, timesteps, features]
    X_train_seq = X_train_seq.reshape(
        (X_train_seq.shape[0], X_train_seq.shape[1], 1)
    )
    X_test_seq = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], 1))
    # Build and train LSTM model
    model = Sequential(
        [
            LSTM(64, input_shape=(X_train_seq.shape[1], 1)),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=20,
        batch_size=32,
        validation_data=(X_test_seq, y_test_seq),
        verbose=0,
    )
    preds_scaled = model.predict(X_test_seq).flatten()
    preds = mms.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    # Transform y_test back to original scale
    y_test = mms.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    # Save the model
    model.save(f"lstm_{ticker}.h5")
    # Align index for LSTM results
    X_test = df.index[-len(preds) :]

    # In LSTM path, use the date index from X_test (time) for results DataFrame later
    # We assign the index to pred_index below

# For RandomForest path, preds and y_test already aligned by index
if model_choice == "RandomForest":
    pred_index = X_test.index
else:
    pred_index = X_test  # for LSTM path, X_test was updated to date index

# Construct results DataFrame
results = pd.DataFrame(
    {
        "actual": y_test if model_choice != "RandomForest" else y_test.values,
        "predicted": preds,
    },
    index=pred_index,
)

# Evaluate model performance
mae = mean_absolute_error(results["actual"], results["predicted"])
rmse = mean_squared_error(results["actual"], results["predicted"], squared=False)
r2 = r2_score(results["actual"], results["predicted"])

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=results.index, y=results["actual"], name="Actual", mode="lines")
    )
    fig.add_trace(
        go.Scatter(
            x=results.index, y=results["predicted"], name="Predicted", mode="lines"
        )
    )
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Price (Adj Close)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Performance")
    st.metric("MAE", f"${mae:,.4f}")
    st.metric("RMSE", f"${rmse:,.4f}")
    st.metric("R²", f"{r2:.4f}")
    st.write("Model saved locally if training completed.")

st.subheader("Detailed results (tail)")
st.dataframe(results.tail(20))

# Provide download of predictions
csv_buffer = io.StringIO()
results.to_csv(csv_buffer)
csv_bytes = csv_buffer.getvalue().encode()
st.download_button(
    "Download results CSV",
    data=csv_bytes,
    file_name=f"{ticker}_predictions.csv",
    mime="text/csv",
)

st.write("---")
st.markdown("**Notes & next steps**")
st.markdown(
    """
    - This does one-step-ahead prediction: predicting next trading day's adjusted close from historical features.
    - For multi-step forecasts, you can iteratively feed predictions back in or train direct multi-step models.
    - Feature ideas: add macro features (indices, interest rates), news sentiment, options flow, fundamentals.
    - To automate live updates, schedule a retrain job and persist the model.
    - This app is educational and not a production trading system.
    """
)

st.markdown(
    "**Caveats**: model performance depends on data quality, selected features, and stationarity. Past performance is not indicative of future results."
)

# Optionally show feature importances for RandomForest
if model_choice == "RandomForest":
    try:
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(
            ascending=False
        ).head(20)
        st.subheader("Top feature importances (Random Forest)")
        st.dataframe(feat_imp)
        fig2 = go.Figure(
            go.Bar(x=feat_imp.values, y=feat_imp.index, orientation="h")
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.write("Could not compute feature importances:", e)
