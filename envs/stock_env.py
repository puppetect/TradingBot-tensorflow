import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import gym
from gym import spaces


class Data:
    def __init__(self, csv_file):
        self.raw = pd.read_csv(csv_file, parse_dates=True, index_col=0)
        self.factors = self.factorize()
        self.prices = self.raw.loc[self.factors.index]

    def factorize(self):
        op = self.raw.open
        cp = self.raw.close
        hp = self.raw.high
        lp = self.raw.low
        rtn = self.daily_return(cp)
        atr = self.average_true_range(hp, lp, cp)
        df = pd.DataFrame({'rtn': rtn, 'atr': atr})
        df = (df - df.mean()) / df.std()
        df.clip(-10., 10., inplace=True)
        df.dropna(inplace=True)
        df['have_position'] = np.zeros(len(df))
        df['duration_trade'] = np.zeros(len(df))
        return df

    @staticmethod
    def daily_return(close, n=1, fillna=False):
        dr = (close / close.shift(n)) - 1
        dr *= 100
        if fillna:
            dr = dr.replace([np.inf, -np.inf], np.nan).fillna(0)
        return pd.Series(dr, name='d_ret')

    @staticmethod
    def average_true_range(high, low, close, n=14, fillna=False):

        cs = close.shift(1)
        tr = high.combine(cs, max) - low.combine(cs, min)

        atr = np.zeros(len(close))
        atr[0] = tr[1::].mean()
        for i in range(1, len(atr)):
            atr[i] = (atr[i - 1] * (n - 1) + tr.iloc[i]) / float(n)

        atr = pd.Series(data=atr, index=tr.index)

        if fillna:
            atr = atr.replace([np.inf, -np.inf], np.nan).fillna(0)

        return pd.Series(atr, name='atr')


class Simulator:
    def __init__(self, data, train_test_split=0.8, trade_period=1000, lots=10000, commission=5):
        self.states = data.factors
        self.prices = data.prices
        self.train_end_index = int(train_test_split * len(self.states))
        self.trade_period = trade_period
        self.min_values = self.states.min(axis=0)
        self.max_values = self.states.max(axis=0)
        self.lots = lots
        self.commission = commission
        self.init_portfolio = {
            'Entry Price': 0,
            'Exit Price': 0,
            'Entry Time': None,
            'Exit Time': None,
            'Profit': 0,
            'Trade Duration': 0,
            'Type': None
        }

    def reset(self, train):
        self.curr_trade_reward = 0
        self.total_reward = 0
        self.total_trade = 0
        self.average_profit_per_trade = 0
        self.have_position = False
        self.journal = []
        self.portfolio = self.init_portfolio
        if train:
            self.offset = 0
            self.end = self.train_end_index - 1
        else:
            self.offset = self.train_end_index
            self.end = len(self.states) - 1
        obs = self.states.iloc[self.offset]
        return obs.values

    def step(self, action):
        prev_close_price = self.prices.close[self.offset]
        self.offset += 1
        curr_open_price = self.prices.open[self.offset]
        curr_close_price = self.prices.close[self.offset]
        curr_timestamp = self.prices.index[self.offset]
        reward = 0

        # buy
        if action == 0:
            if not self.have_position:
                self.portfolio['Entry Price'] = curr_open_price
                self.portfolio['Type'] = 'BUY'
                self.portfolio['Entry Time'] = curr_timestamp
                self.total_trade += 1
                self.have_position = True
            self.portfolio['Trade Duration'] += 1
            reward = (curr_close_price - curr_open_price) * self.lots - self.commission

        # sell
        if action == 1 or self.portfolio['Trade Duration'] >= self.trade_period:
            if self.have_position:
                self.portfolio['Exit Price'] = curr_close_price
                self.portfolio['Exit Time'] = curr_timestamp
                self.portfolio['Profit'] = (curr_close_price - self.portfolio['Entry Price']) * self.lots - self.commission
                self.journal.append(self.portfolio)
                self.portfolio = self.init_portfolio
                self.curr_trade_reward = 0
                self.have_position = False

        # skip
        if action == 2:
            if self.have_position:
                self.portfolio['Trade Duration'] += 1
                reward = (curr_close_price - prev_close_price) * self.lots

        self.curr_trade_reward += reward
        self.total_reward += reward

        if self.total_trade > 0:
            self.average_profit_per_trade = self.total_reward / self.total_trade

        info = {'Aberage reward per trade': self.average_profit_per_trade,
                'Reward for this trade': self.curr_trade_reward,
                'Total reward': self.total_reward}

        next_obs = self.states.iloc[self.offset].values
        next_obs[-2] = self.have_position
        next_obs[-1] = self.portfolio['Trade Duration']

        done = self.offset >= self.end

        return next_obs, self.curr_trade_reward, done, info


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, csv_file, train_test_split, trade_period, lots, commission):
        self.sim = Simulator(Data(csv_file), train_test_split, trade_period, lots, commission)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.sim.min_values, self.sim.max_values, dtype=np.float32)

    def reset(self, train=True):
        obs = self.sim.reset(train)
        return obs

    def step(self, action):
        next_obs, reward, done, info = self.sim.step(action)
        return next_obs, reward, done, info

    def seed(self):
        pass

    def render(self):
        pass


if __name__ == '__main__':
    env = StockEnv('../data/000001_0518.csv', 0.8, 1000, True, 10000, 2.5)
    print(env.observation_space)