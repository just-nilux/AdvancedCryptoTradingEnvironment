import gym
from gym import spaces
import numpy as np
import pandas as pd


class AdvancedCryptoTradingEnvironment(gym.Env):
    """
    An advanced cryptocurrency trading environment for OpenAI Gym
    with a comprehensive trading model.

    State: [Portfolio value, Owned BTC, Owned ETH, Owned USD, Current BTC price, Current ETH price, past n BTC prices, past n ETH prices]

    Actions:
    - 0: HOLD
    - 1: BUY BTC
    - 2: SELL BTC
    - 3: BUY ETH
    - 4: SELL ETH
    """

    def __init__(self, df, n_past_prices=5, max_trades=10, transaction_cost=0.001):
        super(AdvancedCryptoTradingEnvironment, self).__init__()

        self.df = df
        self.n_past_prices = n_past_prices
        self.max_trades = max_trades
        self.transaction_cost = transaction_cost
        self.reward_range = (0, np.inf)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5 + 2 * n_past_prices,))

        self.reset()

    def step(self, action):
        self.current_step += 1
        self._trade(action)

        done = self.current_step > len(self.df.loc[:, 'Open'].values) - 6 or self.current_trade >= self.max_trades
        obs = self._get_observation()
        reward = self._get_reward()
        return obs, reward, done, {}

    def reset(self):
        self.portfolio_value = 10000
        self.btc_held = 0
        self.eth_held = 0
        self.cash_balance = self.portfolio_value
        self.current_step = 0
        self.current_trade = 0
        self.past_btc_prices = np.array(self.df.loc[:self.n_past_prices - 1, 'BTC Price'])
        self.past_eth_prices = np.array(self.df.loc[:self.n_past_prices - 1, 'ETH Price'])
        return self._get_observation()

    def render(self, mode='human', close=False):
        profit = self.portfolio_value - self.cash_balance
        print(f'Step: {self.current_step}, Portfolio Value: {self.portfolio_value}, Profit: {profit}')

    def _get_observation(self):
        btc_prices = np.array(self.df.loc[self.current_step:self.current_step + self.n_past_prices - 1, 'BTC Price'])
        eth_prices = np.array(self.df.loc[self.current_step:self.current_step + self.n_past_prices - 1, 'ETH Price'])
        btc_price = self.df.loc[self.current_step, 'BTC Price']
        eth_price = self.df.loc[self.current_step, 'ETH Price']
        obs = np.array([self.portfolio_value, self.btc_held, self.eth_held, self.cash_balance, btc_price,
                        eth_price] + btc_prices.tolist() + eth_prices.tolist())
        return obs

    def _trade(self, action):
        btc_price = self.df.loc[self.current_step, 'BTC Price']
        eth_price = self.df.loc[self.current_step, 'ETH Price']

        if action == 0:  # Hold
            return
        elif action == 1:  # Buy BTC
            btc_bought = self.cash_balance / btc_price
            self.btc_held += btc_bought
            self.cash_balance -= btc_bought * btc_price
            self.cash_balance -= self.transaction_cost * self.cash_balance
        elif action == 2:  # Sell BTC
            self.cash_balance += self.btc_held * btc_price
            self.btc_held = 0
            self.cash_balance -= self.transaction_cost * self.cash_balance
        elif action == 3:  # Buy ETH
            eth_bought = self.cash_balance / eth_price
            self.eth_held += eth_bought
            self.cash_balance -= eth_bought * eth_price
            self.cash_balance -= self.transaction_cost * self.cash_balance
        elif action == 4:  # Sell ETH
            self.cash_balance += self.eth_held * eth_price
            self.eth_held = 0
            self.cash_balance -= self.transaction_cost * self.cash_balance

        self.portfolio_value = self.cash_balance + self.btc_held * btc_price + self.eth_held * eth_price
        self.past_btc_prices = np.append(self.past_btc_prices[1:], btc_price)
        self.past_eth_prices = np.append(self.past_eth_prices[1:], eth_price)
        self.current_trade += 1

    def _get_reward(self):
        reward = self.portfolio_value - self.cash_balance
        # Reward shaping to encourage profitable trades
        if reward < 0:
            reward *= 0.9
        return reward
