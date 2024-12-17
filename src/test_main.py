'''
The main test file for the project. It mostly tests the integrity. On push to the main branch of the repository, the tests are run on the CI pipeline.
'''
import datetime
import glob
from gymnasium import gym
import logging
import os
import pytest

from dataset.containers import DataSet, DataPoint
from environment import Config, MarketEnv
from util import extract_tickers, extract_from_pickle, get_datasets, get_train_validate_test_datasets

DATA_PATH = os.path.join(".", "data")
DATASET_PATH = os.path.join(DATA_PATH, "AAPL")

logging.basicConfig(level=logging.DEBUG)


def test_dataset_entries_are_chronological():
    data = extract_tickers(DataSet("AAPL"), "AAPL")

    for i in range(len(data) - 1):
        assert 't' in (data.data_points[i]).__dict__.keys() and 't' in data.data_points[i + 1].__dict__.keys()
        assert data.data_points[i] < data.data_points[i + 1]
        assert data.data_points[i].t < data.data_points[i + 1].t


def test_pickles_integrity():
    for f in glob.glob(os.path.join(DATA_PATH, "pickles", "*.pkl")):
        data = extract_from_pickle(f)
        assert isinstance(data, DataSet)
        assert all(isinstance(e, DataPoint) for e in data.data_points)


def test_train_validation_test_split_integrity():
    timestamp1 = datetime.datetime(2024, 2, 15)
    timestamp2 = datetime.datetime(2024, 5, 1)
    train_set, val_set, test_set = get_train_validate_test_datasets(get_datasets())
    for e in train_set:
        assert e <= timestamp1
    for e in val_set:
        assert e <= timestamp2 and e > timestamp1
    for e in test_set:
        assert e > timestamp2


def test_environment_run_does_not_crash():
    env_configs = Config(
        MEAN_RUN_DURATION=100,
        STD_RUN_DURATION=10,
        START_BALANCE=1000,
        MAX_BUY_LIMIT=10,
        CONTINUOUS_MODEL=False,
        TRUNCATION_PENALTY=0,
        STOCK_HOLDING_REWARD=1
    )

    datasets = get_datasets()

    gym.register("MarketEnv-v0", entry_point=MarketEnv)
    env = gym.make("MarketEnv-v0", datasets=datasets, config=env_configs)

    try:
        env.reset(seed=44)
        done = False
        while not done:
            action = 0
            _, _, done, _ = env.step(action)
    except Exception as e:
        assert False, "Run failed with exception: {}".format(e)
