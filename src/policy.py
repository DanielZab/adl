from typing import List, Set
from dataset_container import DataPoint, DataSet
from constants import *


class Action:
    def __init__(self, stock_acquisition: int):
        self.stock_acquisition = stock_acquisition # < 0 for sell, > 0 for buy, 0 for hold

    def __str__(self):
        return f"Action(stock_acquisition={self.stock_acquisition})"
    
    def __repr__(self):
        return str(self)

class State:
    def __init__(self, stock_price: float, balance: float, stocks_owned: int, previous_prices: List[DataPoint]):
        self.stock_price = stock_price
        self.balance = balance
        self.stocks_owned = stocks_owned
        self.previous_prices = previous_prices
        self.actions = self.generate_actions()
        
    def generate_actions(self) -> Set[Action]:
        actions = set()

        potential_max_buy_amount = self.balance // self.stock_price
        max_buy_amount = min(potential_max_buy_amount, MAX_BUY_LIMIT)

        assert isinstance(max_buy_amount, int)

        for i in range(0, max_buy_amount + 1, TRANSACTION_STEP):
            actions.add(Action(i))
        
        actions.add(Action(0))
        return actions
    
    def get_state_value(self) -> float:
        return self.balance + self.stock_price * self.stocks_owned
    
    def __str__(self):
        return f"State(stock_price={self.stock_price}, balance={self.balance}, stocks_owned={self.stocks_owned}, previous_prices={self.previous_prices})"

    def __repr__(self):
        return str(self)