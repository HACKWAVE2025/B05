RL_CONFIG = {
    "episodes": 1000,
    "max_steps_per_episode": 100,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,
    "epsilon_min": 0.01,
    "batch_size": 32,
    "memory_size": 2000,
    "update_target_freq": 10
}

ENV_CONFIG = {
    "initial_capital": 100000,
    "max_portfolio_value": 1000000,
    "min_trade": 100,
    "max_trade": 50000,
    "transaction_fee": 0.001,
    "risk_free_rate": 0.02
}

RISK_PROFILES = {
    1: {"name": "VERY_CONSERVATIVE", "stocks": 0.1, "bonds": 0.6, "cash": 0.3},
    2: {"name": "CONSERVATIVE", "stocks": 0.2, "bonds": 0.5, "cash": 0.3},
    3: {"name": "BALANCED", "stocks": 0.4, "bonds": 0.4, "cash": 0.2},
    4: {"name": "AGGRESSIVE", "stocks": 0.6, "bonds": 0.3, "cash": 0.1},
    5: {"name": "MODERATE/BALANCED", "stocks": 0.5, "bonds": 0.35, "cash": 0.15}
}

print(f"[CONFIG] Risk Appetite 5: {RISK_PROFILES[5]['name']}")
