import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import time
import threading
import json
import os
import pickle


class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.trained_stocks = {}
        self.model_path = "stock_model.h5"
        self.scaler_path = "scaler.pkl"
        self.prediction_cache = {}
        self.cache_time = {}
        self.cache_duration = 3600
        self.lock = threading.Lock()
        self.last_fetch_time = {}
        self.min_request_interval = 1.5
        self.load_model()

    def _rate_limit_wait(self, symbol: str):
        """Rate limiting"""
        with self.lock:
            now = time.time()
            last_time = self.last_fetch_time.get(symbol, 0)
            wait_time = self.min_request_interval - (now - last_time)

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_fetch_time[symbol] = time.time()

    def load_model(self):
        """Load pre-trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                if os.path.exists(self.scaler_path):
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                print("[MODEL] Pre-trained model loaded!")
            except:
                self.model = self.build_model()
        else:
            self.model = self.build_model()

    def build_model(self):
        """Build optimized LSTM model"""
        model = keras.Sequential([
            keras.layers.LSTM(256, activation='relu', input_shape=(60, 8), return_sequences=True),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(128, activation='relu', return_sequences=True),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(64, activation='relu', return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("[MODEL] New optimized LSTM model built!")
        return model

    def calculate_technical_indicators(self, data):
        """Calculate RSI, MACD, Bollinger"""
        close = data['Close'].values

        # RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:]) if len(gain) > 0 else 0
        avg_loss = np.mean(loss[-14:]) if len(loss) > 0 else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50

        # MACD
        ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
        ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
        macd = ema_12 - ema_26

        # Bollinger Bands
        bb_middle = data['Close'].rolling(20).mean().iloc[-1]
        bb_std = data['Close'].rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = (close[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5

        return {
            'rsi': np.clip(rsi / 100, 0, 1),
            'macd': np.tanh(macd),
            'bb_position': np.clip(bb_position, 0, 1)
        }

    def get_stock_data(self, symbol: str, period: str = "1y"):
        """Fetch stock data with rate limiting"""
        try:
            self._rate_limit_wait(symbol)
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty or len(data) < 70:
                print(f"[DATA] {symbol}: Insufficient data ({len(data)} records)")
                return None

            print(f"[DATA] {symbol}: {len(data)} records fetched")
            return data
        except Exception as e:
            print(f"[ERROR] {symbol}: {str(e)}")
            return None

    def prepare_features(self, data):
        """Prepare 8-feature OHLCV + Technical indicators"""
        if data is None or len(data) < 65:
            return None

        close = data['Close'].values
        features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values

        # Normalize OHLCV
        scaled_features = self.scaler.fit_transform(features)

        # Get technical indicators
        indicators = self.calculate_technical_indicators(data)

        X = []
        y = []

        for i in range(60, len(scaled_features)):
            # Combine OHLCV + indicators
            seq = scaled_features[i - 60:i].copy()

            # Add technical indicators to last row
            seq[-1] = np.append(seq[-1], [
                indicators['rsi'],
                indicators['macd'],
                indicators['bb_position']
            ])

            X.append(seq)

            # Label: 1 if next close > current close
            y.append(1 if close[i] > close[i - 1] else 0)

        # Pad sequences to 8 features
        X_padded = []
        for seq in X:
            if seq.shape[1] < 8:
                padding = np.zeros((seq.shape[0], 8 - seq.shape[1]))
                seq = np.hstack([seq, padding])
            X_padded.append(seq[:, :8])

        return np.array(X_padded), np.array(y)

    def train_on_stock(self, symbol: str):
        """Train with early stopping"""
        print(f"[TRAIN] Training on {symbol}...")

        data = self.get_stock_data(symbol)
        if data is None:
            print(f"[ERROR] Could not fetch {symbol}")
            return False

        X, y = self.prepare_features(data)
        if X is None:
            print(f"[ERROR] Could not prepare features for {symbol}")
            return False

        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        )

        # Train
        history = self.model.fit(
            X, y,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluate
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        print(f"[TRAIN] {symbol} - Accuracy: {accuracy * 100:.2f}%")

        self.trained_stocks[symbol] = {
            "accuracy": float(accuracy),
            "timestamp": datetime.now().isoformat()
        }

        # Save model & scaler
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        return accuracy > 0.80

    def bulk_train(self, symbols: list):
        """Train on multiple stocks sequentially"""
        print(f"[TRAIN] Starting bulk training on {len(symbols)} stocks...")
        successful = 0

        for symbol in symbols:
            try:
                if self.train_on_stock(symbol):
                    successful += 1
                time.sleep(2)
            except Exception as e:
                print(f"[TRAIN] Error with {symbol}: {str(e)}")

        print(f"[TRAIN] Bulk training complete! {successful}/{len(symbols)} successful")

    def predict_stock(self, symbol: str):
        """Predict with caching"""
        # Check cache
        if symbol in self.prediction_cache:
            cache_age = time.time() - self.cache_time.get(symbol, 0)
            if cache_age < self.cache_duration:
                return self.prediction_cache[symbol]

        try:
            data = self.get_stock_data(symbol, period="90d")
            if data is None or len(data) < 65:
                return {"symbol": symbol, "prediction": None, "confidence": 0, "signal": "⚠️ ERROR"}

            close = data['Close'].values
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            scaled = self.scaler.fit_transform(features)

            # Prepare last 60 days
            indicators = self.calculate_technical_indicators(data)
            seq = scaled[-60:].copy()
            seq[-1] = np.append(seq[-1], [
                indicators['rsi'],
                indicators['macd'],
                indicators['bb_position']
            ])

            if seq.shape[1] < 8:
                padding = np.zeros((seq.shape[0], 8 - seq.shape[1]))
                seq = np.hstack([seq, padding])

            X = seq[:, :8].reshape(1, 60, 8)

            # Predict
            prediction = self.model.predict(X, verbose=0)[0][0]
            confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)

            current_price = close[-1]
            prev_close = close[-2]
            change = ((current_price - prev_close) / prev_close) * 100

            result = {
                "symbol": symbol,
                "prediction": int(prediction > 0.5),
                "confidence": round(confidence, 4),
                "current_price": float(current_price),
                "change_today": round(change, 2),
                "signal": "📈 BUY" if prediction > 0.5 else "📉 SELL",
                "rsi": round(indicators['rsi'] * 100, 2),
                "macd": round(indicators['macd'], 4),
                "timestamp": datetime.now().isoformat()
            }

            # Cache
            self.prediction_cache[symbol] = result
            self.cache_time[symbol] = time.time()

            return result
        except Exception as e:
            return {"symbol": symbol, "prediction": None, "confidence": 0, "signal": f"❌ {str(e)}"}

    def predict_multiple(self, symbols: list):
        """Predict for multiple stocks"""
        results = []
        for symbol in symbols:
            result = self.predict_stock(symbol)
            if result.get("confidence", 0) > 0.65:  # 65%+ confidence
                results.append(result)

        results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return results


stock_predictor = StockPredictor()
