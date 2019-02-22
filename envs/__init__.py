from gym.envs.registration import register

register(
    id='trading-v0',
    entry_point='envs.stock_env:StockEnv',
    kwargs={'csv_file': 'data/000001_0518.csv',
            'train_test_split': 0.8,
            'trade_period': 3,
            'lots': 100,
            'commission': 2.5}
)
