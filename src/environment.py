'''
This module contains the implementation of the MarketEnv class, which is a custom gym environment for simulating stock market trading. The MarketEnv class is a subclass of the gym.Env class, which is the base class for all gym environments. The MarketEnv class implements the reset and step methods required by the gym.Env class. 

A state is defined by the current stock price, the agents' balance, and stocks owned. The portfolio value is additionally returned at each step for further context. 

An action is a integer that represents the amount of stocks to buy or sell. The action space may be either continuous or discrete, depending on the configuration. If the action space is continuous, the action is rounded to the nearest integer. The action is clipped to the maximum buy limit, which is the maximum number of stocks the agent is allowed to buy or sell at once.
A negative integer means SELLING and a positive integer means BUYING. The agent is not allowed to sell more stocks than it owns, and it is not allowed to buy more stocks than it can afford. Both actions are penalized with a truncation penalty.

The general reward consists of the portfolio value increase in comparison to the previous portfolio value. To motivate the agent to buy stocks, an additional reward is granted based on the amount of stocks in possession.
'''

import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple

from dataset.containers import DataSet, DataPoint


class Config:
    '''
    A container for all the configuration parameters of the MarketEnv class. The configuration parameters are used to customize the behavior of the MarketEnv class:
    - mean and standard deviation of the run duration
    - the starting balance
    - the maximum buy limit, which is how much the agent is allowed to buy or sell at once
    - the continuous model flag which encodes whether the action space of the agent is continuous or discrete. This only makes a difference for the nn and its calculation of pi, as actions are rounded to the nearest integer in the step function
    - the truncation penalty, which decreases the reward if the action, and the stock holding reward.
    '''

    def __init__(self, MEAN_RUN_DURATION=100, STD_RUN_DURATION=10, START_BALANCE=1000, MAX_BUY_LIMIT=None,
                 CONTINUOUS_MODEL=True, TRUNCATION_PENALTY=0, STOCK_HOLDING_REWARD=1):
        self.MEAN_RUN_DURATION = MEAN_RUN_DURATION
        self.STD_RUN_DURATION = STD_RUN_DURATION
        self.START_BALANCE = START_BALANCE
        self.MAX_BUY_LIMIT = MAX_BUY_LIMIT
        self.CONTINUOUS_MODEL = CONTINUOUS_MODEL
        self.TRUNCATION_PENALTY = TRUNCATION_PENALTY
        self.STOCK_HOLDING_REWARD = STOCK_HOLDING_REWARD

    def __str__(self):
        return f"Config(MEAN_RUN_DURATION={self.MEAN_RUN_DURATION}, STD_RUN_DURATION={self.STD_RUN_DURATION}, START_BALANCE={self.START_BALANCE}, MAX_BUY_LIMIT={self.MAX_BUY_LIMIT}, CONTINUOUS_MODEL={self.CONTINUOUS_MODEL}, TRUNCATION_PENALTY={self.TRUNCATION_PENALTY}, STOCK_HOLDING_REWARD={self.STOCK_HOLDING_REWARD})"


class MarketEnv(gym.Env):
    '''
    The main class of the module
    '''

    def __init__(self, datasets: List[DataSet], config: Config):
        '''
        datasets: A list of DataSet objects, each representing a stock
        config: A Config object containing the configuration parameters of the MarketEnv class
        '''

        self.balance = config.START_BALANCE
        self.stocks_owned = 0
        self.current_price = None
        self.datasets = datasets
        self.current_dataset = None
        self.config = config

        self.observation_space = gym.spaces.Dict({
            "current_price": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            # previous prices + current price, starting from the oldest
            "balance": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "stocks_owned": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        })

        if config.CONTINUOUS_MODEL:
            if config.MAX_BUY_LIMIT:
                self.action_space = gym.spaces.Box(low=-config.MAX_BUY_LIMIT, high=config.MAX_BUY_LIMIT, shape=(1,),
                                                   dtype=np.float32)
            else:
                self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(config.MAX_BUY_LIMIT * 2 + 1, start=-config.MAX_BUY_LIMIT)

    def _get_obs(self) -> dict:
        return {"current_price": np.array(self.current_price, dtype=np.float32),
                "balance": np.array(self.balance, dtype=np.float32),
                "stocks_owned": np.array(self.stocks_owned, dtype=np.int32)}

    def _get_info(self) -> dict:
        return {"portfolio_value": self.get_portfolio_value()}

    def get_current_price(self) -> float:
        return self.current_dataset[self.current_increment + self.start_index].price()

    def get_portfolio_value(self) -> float:
        return float(self.current_price * self.stocks_owned + self.balance)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        '''
        Resets the environment to its initial state. The seed parameter is used to seed the random number generator. The options parameter is not used. The reset method returns the initial observation and info of the environment.

        First, a random dataset is selected and a run duration is established. Then, the starting index is chosen.

        Returns: observation, info
        '''

        super().reset(seed=seed)

        randint = self.np_random.integers(0, len(self.datasets))
        self.current_dataset = self.datasets[randint]

        if self.config.MEAN_RUN_DURATION is not None and self.config.STD_RUN_DURATION is not None:
            self.run_duration = round(
                self.np_random.normal(self.config.MEAN_RUN_DURATION, self.config.STD_RUN_DURATION, size=None))
            self.run_duration = min(len(self.current_dataset), self.run_duration)
            assert isinstance(self.run_duration, int)

            # Choose the data point to start from
            self.start_index = self.np_random.integers(0, len(self.current_dataset) - self.run_duration, size=None,
                                                       dtype=np.int32)
        else:
            self.run_duration = len(self.datasets)
            self.start_index = 0

        self.current_increment = 0
        self.current_price = self.get_current_price()

        self.balance = self.config.START_BALANCE
        self.stocks_owned = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def clip_action(self, action) -> int:
        '''
        Clips the action to the maximum buy limit and ensures that the agent does not sell more stocks than it owns or buys more stocks than it can afford.

        Returns: the clipped action
        '''
        if self.config.MAX_BUY_LIMIT:
            action = min(action, self.config.MAX_BUY_LIMIT)

        return int(np.clip(action, -self.stocks_owned, self.balance // self.current_price))

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        '''
        Takes a step in the environment. The action parameter is the amount of stocks to buy or sell.
        
        Returns: observation, reward, done, info.
        '''
        # Round action if the model is not continuous, else ensure it is a valid action
        if not self.config.CONTINUOUS_MODEL:
            action = int(action)
            assert self.action_space.contains(action), f"Action {action} is not in the action space {self.action_space}"
        else:
            action = int(np.round(action, decimals=0))

        # Clip the action to the maximum buy limit
        old_action = action
        action = self.clip_action(action)
        truncated = bool(old_action != action)

        # Update balance and stocks owned
        self.balance -= float(action) * float(self.current_price)
        self.stocks_owned += action

        old_portfolio_value = self.get_portfolio_value()

        # Move to the next data point
        self.current_increment += 1
        self.current_price = self.get_current_price()

        observation = self._get_obs()
        info = self._get_info()

        # Calculate penalty and additional stock holding reward
        truncation_penalty = self.config.TRUNCATION_PENALTY if truncated else 0
        stock_holding_reward = self.config.STOCK_HOLDING_REWARD * self.stocks_owned

        # Calculate reward by the change in portfolio value
        reward = self.get_portfolio_value() - old_portfolio_value - truncation_penalty + stock_holding_reward

        # Done if we reached the end of the data set
        done = self.current_increment >= self.run_duration

        return observation, reward, done, info
