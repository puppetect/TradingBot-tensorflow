import gym
import envs
from agents.dqn import TradingDQN
from agents.policy import FFPolicy


def dqn():
    env = gym.make('trading-v0')
    model = TradingDQN(FFPolicy, env, tensorboard_log='saves/log')
    model.learn(total_timesteps=5000000, test_interval=1)


if __name__ == '__main__':
    dqn()
