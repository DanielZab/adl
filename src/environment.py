from typing import Optional, List
import numpy as np
import gymnasium as gym
from dataset.containers import DataSet, DataPoint

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
    
class Config:
    def __init__(self, MEAN_RUN_DURATION=100, STD_RUN_DURATION=10, START_BALANCE=1000, MAX_BUY_LIMIT=None, CONTINUOUS_MODEL=True, TRUNCATION_PENALTY=0, STOCK_HOLDING_REWARD=1):
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

    def __init__(self, datasets: List[DataSet], config: Config):
        
        self.balance = config.START_BALANCE
        self.stocks_owned = 0
        self.current_price = None
        self.datasets = datasets
        self.current_dataset = None
        self.config = config

        self.observation_space = gym.spaces.Dict({
                "current_price": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # previous prices + current price, starting from the oldest
                "balance": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "stocks_owned": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
            }) 

        if config.CONTINUOUS_MODEL:
            if config.MAX_BUY_LIMIT:
                self.action_space = gym.spaces.Box(low=-config.MAX_BUY_LIMIT, high=config.MAX_BUY_LIMIT, shape=(1,), dtype=np.float32)
            else: 
                self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(config.MAX_BUY_LIMIT * 2 + 1, start=-config.MAX_BUY_LIMIT)

    def _get_obs(self):
        return {"current_price": np.array(self.current_price, dtype=np.float32), "balance": np.array(self.balance, dtype=np.float32), "stocks_owned": np.array(self.stocks_owned, dtype=np.int32)}
    
    def _get_info(self):
        return {"portfolio_value": self.get_portfolio_value()}
    
    def get_current_price(self):
        return self.current_dataset[self.current_increment + self.start_index].price()
    
    def get_portfolio_value(self):
        return float(self.current_price * self.stocks_owned + self.balance)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        randint = self.np_random.integers(0, len(self.datasets))
        self.current_dataset = self.datasets[randint]

        if self.config.MEAN_RUN_DURATION is not None and self.config.STD_RUN_DURATION is not None:
            self.run_duration = round(self.np_random.normal(self.config.MEAN_RUN_DURATION, self.config.STD_RUN_DURATION, size=None))
            self.run_duration = min(len(self.current_dataset), self.run_duration)
            assert isinstance(self.run_duration, int)

            # Choose the data point to start from
            self.start_index = self.np_random.integers(0, len(self.current_dataset) - self.run_duration, size=None, dtype=np.int32)
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
    
    def clip_action(self, action):
        if self.config.MAX_BUY_LIMIT:
            action = min(action, self.config.MAX_BUY_LIMIT)
        
        return int(np.clip(action, -self.stocks_owned, self.balance // self.current_price))

    def step(self, action):
        if not self.config.CONTINUOUS_MODEL:
            assert self.action_space.contains(int(action)), f"Action {action} is not in the action space {self.action_space}"
        else:
            action = int(np.round(action, decimals=0))

        # clip the action to the maximum buy limit
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

        truncation_penalty = self.config.TRUNCATION_PENALTY if truncated else 0
        stock_holding_reward = self.config.STOCK_HOLDING_REWARD * self.stocks_owned

        # Reward is the change in portfolio value
        reward = self.get_portfolio_value() - old_portfolio_value - truncation_penalty + stock_holding_reward

        # Done if we reached the end of the data set
        done = self.current_increment >= self.run_duration

        return observation, reward, done, info
