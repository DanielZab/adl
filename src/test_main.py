import pytest
import logging, os, glob
import datetime

from dataset_container import DataSet, DataPoint
from util import extract_tickers, extract_from_pickle


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