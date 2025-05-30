import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import requests

def scrape_nasdaq100_symbols():
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    try:
        tables = pd.read_html(url)
        # Find the correct table with tickers (usually the first one or second one)
        for table in tables:
            if "Ticker" in table.columns or "Ticker symbol" in table.columns:
                ticker_col = "Ticker" if "Ticker" in table.columns else "Ticker symbol"
                symbols = table[ticker_col].tolist()
                # Clean symbols (remove any suffixes or whitespace)
                symbols = [sym.strip().replace('.', '-') for sym in symbols]
                return symbols
        print("Could not find ticker table on NASDAQ-100 Wikipedia page.")
        return []
    except Exception as e:
        print(f"Error scraping NASDAQ-100 symbols: {e}")
        return []

def get_crypto_symbols():
    # Add popular cryptos manually
    return [
        "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD",
        "DOGE-USD", "MATIC-USD", "SOL-USD", "DOT-USD", "LTC-USD"
    ]

def get_features(df):
    try:
        df = df.copy()
        df['SMA_10'] = SMAIndicator(df['Close'], window=10).sma_indicator().squeeze()
        df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator().squeeze()
        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error generating features: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def train_model(ticker):
    print(f"Downloading data for {ticker}...")
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty:
            print(f"No data found for {ticker}, skipping.")
            return
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return

    df = get_features(df)
    if df.empty:
        print(f"Insufficient data to train model for {ticker}, skipping.")
        return

    X = df[['SMA_10', 'SMA_50', 'Return']]
    y = df['Target']

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print(f"Training model for {ticker}...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model accuracy for {ticker}: {acc * 100:.2f}%")

        model_filename = f"models/model_{ticker.replace('-', '_')}.pkl"
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}\n")

    except Exception as e:
        print(f"Error training or saving model for {ticker}: {e}")

if __name__ == "__main__":
    symbols_file = "symbols.csv"

    # Load or scrape symbols
    if os.path.exists(symbols_file):
        try:
            tickers_df = pd.read_csv(symbols_file)
            tickers = tickers_df['symbol'].tolist()
            print(f"Loaded {len(tickers)} tickers from {symbols_file}.")
        except Exception as e:
            print(f"Error reading {symbols_file}: {e}")
            tickers = []
    else:
        print(f"{symbols_file} not found. Scraping NASDAQ-100 symbols...")
        tickers = scrape_nasdaq100_symbols()
        cryptos = get_crypto_symbols()
        tickers.extend(cryptos)

        if tickers:
            try:
                pd.DataFrame({"symbol": tickers}).to_csv(symbols_file, index=False)
                print(f"Saved {len(tickers)} tickers to {symbols_file}.")
            except Exception as e:
                print(f"Error saving symbols to {symbols_file}: {e}")
        else:
            print("No tickers found. Exiting.")
            exit(1)

    if not tickers:
        print("No tickers to train on. Exiting.")
        exit(1)

    for ticker in tickers:
        train_model(ticker)
