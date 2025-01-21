'''
This file is the backend implementation of the stock trading demo. It uses FastAPI to create a WebSocket server that sends stock price data to the frontend. The backend also uses PyTorch to load a pre-trained model and thus makes trading decisions based on the stock price data.
'''
import os
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from typing import List
import asyncio
import random
import requests
from bs4 import BeautifulSoup
from dataset.containers import DataSet
from datetime import datetime
from util import get_datasets, get_train_validate_test_datasets
import ppo as ppo

import numpy as np

SLEEP_TIME = 10

# Load the datasets
datasets = get_datasets()
train_sets, validation_sets, test_sets = get_train_validate_test_datasets(datasets)

class SetIterator:
    '''A helper iterator class for iterating the dataset'''
    def __init__(self, data: DataSet):
        self.dataset = data
        self.name = data.name
        self.iter = self.generator()
        
    def generator(self):
        for x in self.dataset:
            yield x.price()
            
    def __next__(self):
        return next(self.iter)
        

test_iters = [SetIterator(e) for e in test_sets]

app = FastAPI()
MODEL_PATH = os.path.join("..", "models","saved_models", "final", "model-epoch=958-avg_reward=0.04050.ckpt")

def do_step(model: ppo.PPO, hn, balance, stock_price, stocks_holding):
    '''
    Performs a single step of the model. The model is used to make a decision based on the current state.
    
    model: the model to use for making the decision
    hn: hidden state of the model
    balance: current balance of the agent
    stock_price: current price of the stock
    stocks_holding: number of stocks the agent currently owns
    '''
    model.eval()
    state = ({
            "current_price": np.array(stock_price, dtype=np.float32),
            "balance": np.array(balance, dtype=np.float32),
            "stocks_owned": np.array(stocks_holding, dtype=np.int32),
        }, 
             {
                 "portfolio_value": np.array(balance + stocks_holding * stock_price, dtype=np.float32)
             })
    if hn == None:
        hn = model.actor.actor_net.init_hidden(1)
    return model.make_move(state, hn, "cpu")


def get_stock_price(ticker_symbol):
    """Extract the next stock price for a given ticker symbol"""
    
    
    return next([e for e in test_iters if e.name == ticker_symbol][0])



@app.websocket("/ws/{ticker}/{start_price}/{speed}")
async def websocket_endpoint(websocket: WebSocket, ticker: str, start_price: str, speed: float):
    '''
    The main WebSocket endpoint for the backend. This function is called when a new WebSocket connection is established.
    This function sends stock price data and the agents' decisions to the frontend periodically.
    
    websocket: the WebSocket object
    ticker: the ticker symbol of the stock
    start_price: the initial balance of the agent
    speed: the speed at which the data should be sent to the frontend
    '''
    
    await websocket.accept()
    balance = float(start_price)
    stocks_holding = 0
    model = ppo.PPO.load_from_checkpoint(MODEL_PATH, env=None)
    hn = None
    warmup = 5
    try:
        while True:
            current_price = get_stock_price(ticker)
            if current_price is None:
                raise WebSocketDisconnect(reason="Current price could not be fetched")
            action, hn = do_step(model, hn, balance, current_price, stocks_holding)
            action = int(np.clip(action, -stocks_holding, balance // current_price))
            if warmup > 0:
                warmup -= 1
            else:
                balance -= float(action) * float(current_price)
                stocks_holding += action
            data = {
                "balance": balance,
                "stocks_owned": stocks_holding,
                "action": action,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price
            }
            await websocket.send_json(data)
            await asyncio.sleep(SLEEP_TIME/speed)
    except WebSocketDisconnect:
        print("Disconnected")
