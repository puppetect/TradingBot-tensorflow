from gym.envs.registration import register

register(
    id='stock-v0',
    entry_point='envs.stock_env:StockEnv',
)
