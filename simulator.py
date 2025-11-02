import numpy as np
from datetime import datetime


class PortfolioSimulator:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_value = initial_capital
        self.history = []
        self.trades = []

    def record_state(self, allocation, roi, risk):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "value": self.current_value,
            "allocation": allocation,
            "roi": roi,
            "risk": risk
        })

    def execute_trade(self, symbol, quantity, price, side):
        trade_value = quantity * price
        fee = trade_value * 0.001

        if side == "BUY":
            self.current_value -= (trade_value + fee)
        else:
            self.current_value += (trade_value - fee)

        self.trades.append({
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "side": side,
            "timestamp": datetime.now().isoformat()
        })

    def get_performance(self):
        if not self.history:
            return {"roi": 0, "max_drawdown": 0}

        values = [h["value"] for h in self.history]
        roi = (values[-1] - self.initial_capital) / self.initial_capital
        max_drawdown = (min(values) - max(values)) / max(values) if max(values) > 0 else 0

        return {"roi": roi, "max_drawdown": max_drawdown}
