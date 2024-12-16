from typing import Optional
import numpy as np
import gymnasium as gym
from dataset_container import DataSet, DataPoint
from constants import *

class MyQueue:
    def __init__(self, max_size, start_value = 0):
        self.max_size = max_size
        self.queue = [start_value] * max_size

    def append(self, element):
        assert len(self.queue) <= self.max_size
        if len(self.queue) == self.max_size:
            self.queue.pop(0)
        self.queue.append(element)

    def get(self):
        return np.array(self.queue, dtype=np.float32)

    def __len__(self):
        return len(self.queue)
    
    def __str__(self):
        return str(self.queue)
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, key):
        return self.queue[key]

class MarketEnv(gym.Env):

    def __init__(self, data: DataSet, config: dict):
        config_keys = ["MEAN_RUN_DURATION", "STD_RUN_DURATION", "START_BALANCE", "PREVIOUS_DATA_POINTS_AMOUNT", "MAX_BUY_LIMIT"]
        assert all(k in config for k in config_keys)

        self.MEAN_RUN_DURATION = config["MEAN_RUN_DURATION"]
        self.STD_RUN_DURATION = config["STD_RUN_DURATION"]
        self.START_BALANCE = config["START_BALANCE"]
        self.PREVIOUS_DATA_POINTS_AMOUNT = config["PREVIOUS_DATA_POINTS_AMOUNT"]
        self.MAX_BUY_LIMIT = config["MAX_BUY_LIMIT"]
        
        self.balance = self.START_BALANCE
        self.stocks_owned = 0
        self.previous_prices = MyQueue(self.PREVIOUS_DATA_POINTS_AMOUNT)
        self.current_price = None
        self.dataset = data

        self.observation_space = gym.spaces.Dict({
                "current_price": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # previous prices + current price, starting from the oldest
                "balance": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "stocks_owned": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
                "previous_prices": gym.spaces.Box(low=0, high=np.inf, shape=(self.PREVIOUS_DATA_POINTS_AMOUNT,), dtype=np.float32)
            }) 

        if CONTINUOUS_MODEL:
            if self.MAX_BUY_LIMIT:
                self.action_space = gym.spaces.Box(low=-self.MAX_BUY_LIMIT, high=self.MAX_BUY_LIMIT, shape=(1,), dtype=np.float32)
            else: 
                self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(1)

    def _get_obs(self):
        return {"current_price": np.array(self.current_price, dtype=np.float32), "balance": np.array(self.balance, dtype=np.float32), "stocks_owned": np.array(self.stocks_owned, dtype=np.int32), "previous_prices": self.previous_prices.get()}
    
    def _get_info(self):
        return {"portfolio_value": self.get_portfolio_value()}
    
    def get_current_price(self):
        return self.dataset[self.current_increment + self.start_index].price()
    
    def get_portfolio_value(self):
        return self.current_price * self.stocks_owned + self.balance
    
    def define_action_space(self):
        low = int(-self.stocks_owned)
        high = int(self.balance // self.current_price)
        self.action_space = gym.spaces.Discrete(int(high - low + 1), start=int(low))
        assert np.array(low, dtype=np.int32) in self.action_space and np.array(high, dtype=np.int32) in self.action_space
        print(low, high, self.action_space)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.MEAN_RUN_DURATION and self.STD_RUN_DURATION:
            self.run_duration = round(self.np_random.normal(self.MEAN_RUN_DURATION, self.STD_RUN_DURATION, size=None))

            assert isinstance(self.run_duration, int)

            # Choose the data point to start from
            self.start_index = self.np_random.integers(0, len(self.dataset) - self.run_duration, size=None, dtype=np.int32)
        else:
            self.run_duration = len(self.dataset)
            self.start_index = 0

        self.current_increment = 0
        self.current_price = self.get_current_price()

        self.balance = self.START_BALANCE
        self.stocks_owned = 0
        self.previous_prices = MyQueue(self.PREVIOUS_DATA_POINTS_AMOUNT, self.current_price)

        if not CONTINUOUS_MODEL:
            # Redefine the action space
            self.define_action_space()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def clip_action(self, action):
        if self.MAX_BUY_LIMIT:
            action = min(action, self.MAX_BUY_LIMIT)
        
        return np.clip(action, -self.stocks_owned, self.balance // self.current_price)

    def step(self, action):
        if not CONTINUOUS_MODEL:
            assert self.action_space.contains(action)
        else:
            action = int(action)

        # clip the action to the maximum buy limit
        old_action = action
        action = self.clip_action(action)
        truncated = bool(old_action != action)

        assert self.MAX_BUY_LIMIT or not truncated

        # Update balance and stocks owned
        self.balance -= action * self.current_price
        self.stocks_owned += action

        old_portfolio_value = self.get_portfolio_value()

        # Update previous prices
        self.previous_prices.append(self.current_price)

        # Move to the next data point
        self.current_increment += 1
        self.current_price = self.get_current_price()

        observation = self._get_obs()
        info = self._get_info()

        truncation_penalty = TRUNCATION_PENALTY if truncated else 0

        # Reward is the change in portfolio value
        reward = self.get_portfolio_value() - old_portfolio_value - truncation_penalty

        # Done if we reached the end of the data set
        done = self.current_increment >= self.run_duration

        if not CONTINUOUS_MODEL:
            # Redefine the action space
            self.define_action_space()

        return observation, reward, done, info
