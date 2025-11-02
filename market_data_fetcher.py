import yfinance as yf
from datetime import datetime, timedelta
import time
import threading


class MarketDataFetcher:
    def __init__(self):
        self.categories = {
            "staking": ["MSFT", "AAPL", "JNJ", "KO", "PG", "UL", "PEP", "MO"],
            "growth": ["NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD", "NFLX", "SHOP"],
            "dividend": ["JPM", "PG", "JNJ", "KO", "MCD", "WMT", "PEP", "VZ"],
            "crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "XRP-USD", "DOGE-USD"]
        }
        self.price_cache = {}
        self.cache_time = {}
        self.cache_duration = 3600  # 1 hour cache
        self.lock = threading.Lock()
        self.last_fetch_time = {}
        self.min_request_interval = 2  # 2 seconds between requests

    def _rate_limit_wait(self, symbol: str):
        """Rate limiting - wait if needed"""
        with self.lock:
            now = time.time()
            last_time = self.last_fetch_time.get(symbol, 0)
            wait_time = self.min_request_interval - (now - last_time)

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_fetch_time[symbol] = time.time()

    def get_category_stocks(self, category: str):
        return self.categories.get(category, [])

    def recommend_stocks_for_category(self, category: str, num: int = 6):
        stocks = self.get_category_stocks(category)
        return stocks[:num]

    def get_stock_price(self, symbol: str, use_cache=True):
        try:
            # Check cache first
            if use_cache and symbol in self.price_cache:
                cache_age = time.time() - self.cache_time.get(symbol, 0)
                if cache_age < self.cache_duration:
                    return self.price_cache[symbol]

            # Rate limit
            self._rate_limit_wait(symbol)

            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")

            if hist.empty or len(hist) < 2:
                return {"symbol": symbol, "price": 0, "change": 0, "change_pct": 0.0, "error": "No data"}

            current = float(hist["Close"].iloc[-1])
            previous = float(hist["Close"].iloc[-2])
            change = current - previous
            change_pct = (change / previous * 100) if previous != 0 else 0.0

            result = {
                "symbol": symbol,
                "price": round(current, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 4),
                "day_high": round(hist["High"].iloc[-1], 2),
                "day_low": round(hist["Low"].iloc[-1], 2),
                "volume": int(hist["Volume"].iloc[-1])
            }

            # Cache it
            self.price_cache[symbol] = result
            self.cache_time[symbol] = time.time()

            print(f"[FETCH] {symbol} - Price: ${current:.2f}, Change: {change_pct:.2f}%")
            return result

        except Exception as e:
            print(f"[ERROR] {symbol}: {str(e)}")
            return {"symbol": symbol, "price": 0, "change": 0, "change_pct": 0.0, "error": str(e)}

    def get_stock_prices(self, symbols):
        """Fetch prices with rate limiting"""
        results = {}
        for symbol in symbols:
            result = self.get_stock_price(symbol, use_cache=True)
            results[symbol] = result
            time.sleep(0.3)  # Small delay between calls
        return results

    def find_stocks_matching_roi(self, target_roi: float):
        """Find stocks matching ROI (use cached data ONLY - NO fresh fetches)"""
        all_stocks = ["MSFT", "AAPL", "GOOGL", "TSLA", "NVDA", "META"]

        matching = []
        for symbol in all_stocks:
            # Use ONLY cached data - no fresh fetches!
            if symbol in self.price_cache:
                price_data = self.price_cache[symbol]
                if price_data.get("price", 0) > 0:
                    matching.append({
                        "symbol": symbol,
                        "change_pct": price_data["change_pct"],
                        "price": price_data["price"]
                    })

        # If no cached data, return empty list (don't fetch!)
        if not matching:
            return []

        matching.sort(key=lambda x: abs(x["change_pct"] - target_roi))
        return matching[:5]


# CREATE INSTANCE
market_data_fetcher = MarketDataFetcher()
