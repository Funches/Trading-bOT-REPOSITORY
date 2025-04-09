import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import time
import datetime
# Initialize NewsAPI client
newsapi = NewsApiClient(api_key='b309cd0d-0724-42fd-9368-56971ec1784d')
# List of reliable news sources
reliable_sources = [
    'bloomberg', 'marketwatch', 'seeking-alpha', 'reuters', 'investors-business-daily'
]
# Fetch market-related news
def get_market_news():
    all_articles = newsapi.get_everything(
        q="stocks OR market OR earnings OR merger OR breakout",  # Keywords to track
        sources="bloomberg,marketwatch,seeking-alpha",  # Limit to financial sources
        language="en",
        sort_by="publishedAt",  # Sort by latest
        page_size=5
    )
    return all_articles['articles']
STOCK = "SPY"
LOOKBACK_DAYS = 60
DISCORD_WEBHOOK = "https://discord.gg/S5dm6nZU"

def fetch_data():
    df = yf.download(STOCK, period=f"{LOOKBACK_DAYS}d", interval="15m")
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    df["rsi"] = RSIIndicator(close=df["Close"]).rsi()
    df["macd"] = MACD(close=df["Close"]).macd_diff()
    df["sma"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["obv"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df.dropna(inplace=True)
    return df

def get_news_sentiment():
    url = f"https://www.google.com/search?q={STOCK}+stock+news&hl=en"
    headers = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")
    headlines = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
    
    if not headlines:
        return 0  # Neutral if no news

    sentiment = 0
    for h in headlines[:5]:
        text = h.get_text()
        if "drop" in text.lower() or "lawsuit" in text.lower() or "misses" in text.lower():
            sentiment -= 1
        elif "soars" in text.lower() or "beats" in text.lower() or "rally" in text.lower():
            sentiment += 1
    return sentiment

def train_ml_model(df):
    features = df[["rsi", "macd", "sma", "obv"]]
    target = (df["Close"].shift(-1) > df["Close"]).astype(int)  # 1 if price goes up next period
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    return clf, scaler

def make_prediction(model, scaler, df):
    latest = df.iloc[-1][["rsi", "macd", "sma", "obv"]].values.reshape(1, -1)
    scaled = scaler.transform(latest)
    return model.predict(scaled)[0]

def send_alert(message):
    if DISCORD_WEBHOOK != "https://discord.gg/S5dm6nZU":
        requests.post(DISCORD_WEBHOOK, json={"content": message})

def main():
    print(f"✅ [{datetime.datetime.now()}] Bot Started")
    while True:
        try:
            df = fetch_data()
            df = add_technical_indicators(df)
            news_score = get_news_sentiment()
            model, scaler = train_ml_model(df)
            signal = make_prediction(model, scaler, df)

            action = "🟢 BUY" if signal == 1 and news_score >= 0 else "🔴 SELL"
            price = df.iloc[-1]["Close"]
            print(f"[{datetime.datetime.now()}] Signal: {action} @ {price:.2f} | News score: {news_score}")
            send_alert(f"{action} {STOCK} @ {price:.2f} | News score: {news_score}")
        
        except Exception as e:
            print("⚠️ Error:", e)
        
        time.sleep(60 * 15)  # Wait 15 minutes

if __name__ == "__main__":
    main()
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from polygon import RESTClient
from bs4 import BeautifulSoup
from newspaper import Article
import time
import datetime

# Polygon API Key
API_KEY = 'your_polygon_api_key'
client = RESTClient(API_KEY)

# Stock you're trading
STOCK = "SPY"
LOOKBACK_DAYS = 60
DISCORD_WEBHOOK = "YOUR_DISCORD_WEBHOOK"

def fetch_data():
    # Fetch historical data from Polygon API
    try:
        # Fetch aggregated data for 1 minute (for backtesting or live data)
        aggs = client.get_aggs(STOCK, 1, "minute", limit=LOOKBACK_DAYS)
        df = pd.DataFrame(aggs)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print("Error fetching data from Polygon:", e)
        return None

def add_technical_indicators(df):
    # Add RSI, MACD, SMA, OBV technical indicators
    df["rsi"] = RSIIndicator(close=df["close"]).rsi()
    df["macd"] = MACD(close=df["close"]).macd_diff()
    df["sma"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    df.dropna(inplace=True)
    return df

def get_news_sentiment():
    url = f"https://www.google.com/search?q={STOCK}+stock+news&hl=en"
    headers = {"User-Agent": "Mozilla/5.0"}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")
    headlines = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
    
    if not headlines:
        return 0  # Neutral if no news

    sentiment = 0
    for h in headlines[:5]:
        text = h.get_text()
        if "drop" in text.lower() or "lawsuit" in text.lower() or "misses" in text.lower():
            sentiment -= 1
        elif "soars" in text.lower() or "beats" in text.lower() or "rally" in text.lower():
            sentiment += 1
    return sentiment

def train_ml_model(df):
    features = df[["rsi", "macd", "sma", "obv"]]
    target = (df["close"].shift(-1) > df["close"]).astype(int)  # 1 if price goes up next period
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    return clf, scaler

def make_prediction(model, scaler, df):
    latest = df.iloc[-1][["rsi", "macd", "sma", "obv"]].values.reshape(1, -1)
    scaled = scaler.transform(latest)
    return model.predict(scaled)[0]

def send_alert(message):
    if DISCORD_WEBHOOK != "YOUR_DISCORD_WEBHOOK":
        requests.post(DISCORD_WEBHOOK, json={"content": message})

def main():
    print(f"✅ [{datetime.datetime.now()}] Bot Started")
    while True:
        try:
            df = fetch_data()
            if df is None:
                continue
            
            df = add_technical_indicators(df)
            news_score = get_news_sentiment()
            model, scaler = train_ml_model(df)
            signal = make_prediction(model, scaler, df)

            action = "🟢 BUY" if signal == 1 and news_score >= 0 else "🔴 SELL"
            price = df.iloc[-1]["close"]
            print(f"[{datetime.datetime.now()}] Signal: {action} @ {price:.2f} | News score: {news_score}")
            send_alert(f"{action} {STOCK} @ {price:.2f} | News score: {news_score}")
        
        except Exception as e:
            print("⚠️ Error:", e)
        
        time.sleep(60 * 1)  # Wait 1 minute (change to 15 minutes for 15m chart)

if __name__ == "__main__":
    main()
