import numpy as np
from datetime import datetime


class PortfolioCalculator:
    def __init__(self):
        self.history = []

    def add_state(self, value, allocation, roi, risk):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "allocation": allocation,
            "roi": roi,
            "risk": risk
        })


class DeFiEnvironment:
    def __init__(self, initial_capital=100000, risk_appetite=5):
        self.initial_capital = initial_capital
        self.risk_appetite = risk_appetite
        self.portfolio_value = initial_capital
        self.allocation = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.portfolio_calc = PortfolioCalculator()
        self.step_count = 0
        self.max_steps = 100

        self.portfolio_calc.add_state(
            value=self.portfolio_value,
            allocation=self.allocation,
            roi=0.0,
            risk=50
        )

    def reset(self):
        self.portfolio_value = self.initial_capital
        self.allocation = [0.2, 0.2, 0.2, 0.2, 0.2]
        self.step_count = 0
        self.portfolio_calc.history = []
        self.portfolio_calc.add_state(
            value=self.portfolio_value,
            allocation=self.allocation,
            roi=0.0,
            risk=50
        )
        return np.array(self.allocation)

    def step(self, action):
        self.step_count += 1

        price_change = np.random.normal(1.0, 0.02)
        self.portfolio_value *= price_change

        roi = (self.portfolio_value - self.initial_capital) / self.initial_capital
        reward = roi * 100

        self.portfolio_calc.add_state(
            value=self.portfolio_value,
            allocation=self.allocation,
            roi=roi,
            risk=50
        )

        done = self.step_count >= self.max_steps
        info = {"roi": roi}

        return np.array(self.allocation), reward, done, info
