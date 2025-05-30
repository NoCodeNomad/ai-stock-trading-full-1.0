import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def get_features(df):
    df['SMA_10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
    df['SMA_30'] = SMAIndicator(df['Close'], window=30).sma_indicator()
    df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bollinger = BollingerBands(df['Close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['Volume'] = df['Volume']
    df = df.dropna()
    return df

def label_data(df):
    df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
    df['Label'] = 0
    df.loc[df['Future_Return'] > 0.02, 'Label'] = 1
    df.loc[df['Future_Return'] < -0.02, 'Label'] = -1
    return df.dropna()

def train_model(ticker='AAPL'):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period="2y")
    df = get_features(df)
    df = label_data(df)
    features = df[['SMA_10', 'SMA_30', 'RSI_14', 'MACD', 'MACD_signal', 'bb_high', 'bb_low', 'Volume']]
    labels = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("Model accuracy:", model.score(X_test, y_test))
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_model()
