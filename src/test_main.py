import pytest
import logging, os, glob
import datetime

from dataset.containers import DataSet, DataPoint
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

def test_train_validation_test_split():
    timestamp1 = datetime.datetime(2024, 2, 15)
    timestamp2 = datetime.datetime(2024, 5, 1)
    train_set, val_set, test_set = get_train_validate_test_datasets(get_datasets())
    for e in train_set:
        assert e <= timestamp1
    for e in val_set:
        assert e <= timestamp2 and e > timestamp1
    for e in test_set:
        assert e > timestamp2
    