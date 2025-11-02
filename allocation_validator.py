import numpy as np
from typing import Dict, List, Tuple


class AllocationValidator:
    def __init__(self, accuracy_threshold: float = 0.85):
        self.accuracy_threshold = accuracy_threshold
        self.tolerance = 1 - accuracy_threshold

    def validate_allocation(self, target_allocation: np.ndarray, actual_allocation: np.ndarray) -> Dict:
        target = np.array(target_allocation, dtype=float)
        actual = np.array(actual_allocation, dtype=float)
        deviations = np.abs(actual - target)
        tolerance_bands = target * self.tolerance
        within_tolerance = deviations <= tolerance_bands
        accuracy = np.mean(within_tolerance)
        is_valid = accuracy >= self.accuracy_threshold

        return {
            "is_valid": bool(is_valid),
            "accuracy_pct": float(round(accuracy * 100, 2)),
            "target_allocation": [float(x) for x in target],
            "actual_allocation": [float(x) for x in actual],
            "deviations": [float(round(d, 4)) for d in deviations],
            "within_tolerance": [bool(x) for x in within_tolerance],
            "rebalance_needed": not bool(is_valid),
            "tolerance_bands": [float(round(t, 4)) for t in tolerance_bands]
        }

    def calculate_rebalance(self, target_allocation: np.ndarray, current_allocation: np.ndarray,
                            portfolio_value: float) -> Dict:
        target = np.array(target_allocation, dtype=float)
        current = np.array(current_allocation, dtype=float)
        ideal_values = target * portfolio_value
        current_values = current * portfolio_value
        rebalance_amounts = ideal_values - current_values
        buys = {}
        sells = {}

        for i, amount in enumerate(rebalance_amounts):
            if amount > 0:
                buys[i] = float(round(amount, 2))
            elif amount < 0:
                sells[i] = float(round(abs(amount), 2))

        return {
            "target_allocation": [float(x) for x in target],
            "current_allocation": [float(x) for x in current],
            "portfolio_value": float(portfolio_value),
            "ideal_values": [float(round(v, 2)) for v in ideal_values],
            "current_values": [float(round(v, 2)) for v in current_values],
            "rebalance_amounts": [float(round(v, 2)) for v in rebalance_amounts],
            "total_to_buy": float(round(sum(buys.values()), 2)),
            "total_to_sell": float(round(sum(sells.values()), 2)),
            "buys": buys,
            "sells": sells
        }


allocation_validator = AllocationValidator(accuracy_threshold=0.85)
