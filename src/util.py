import glob, os, json, pickle
import datetime
from typing import List
import logging
from constants import PICKLES_PATH, DATA_PATH
from dataset.containers import DataSet

def save_to_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def extract_from_pickle(path: str) -> DataSet:
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_from_json(container: DataSet, path: str):
    with open(path, "rb") as f:
        data = json.load(f)
    
    container.insert(data["results"])

def extract_tickers(container: DataSet, symbol: str) -> DataSet:
    files = glob.glob(os.path.join(DATA_PATH, symbol, "*.json"))
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    for f in sorted_files:
        extract_from_json(container, f)
    return container

def get_datasets():

    datasets = list(glob.glob(os.path.join(PICKLES_PATH, "*.pkl")))

    datasets: List[DataSet] = list([extract_from_pickle(dataset) for dataset in datasets])

    assert all(isinstance(dataset, DataSet) for dataset in datasets)
    return datasets

def get_train_validate_test_datasets(datasets: List[DataSet], timestamp1: datetime.datetime = datetime.datetime(2024, 1, 1), timestamp2: datetime.datetime = datetime.datetime(2024, 7, 1)):
    train_sets = []
    validation_sets = []
    test_sets = []

    for dataset in datasets:
        train_set, validation_set, test_set = dataset.train_validation_test_split(timestamp1, timestamp2)
        train_sets.append(train_set)
        validation_sets.append(validation_set)
        test_sets.append(test_set)
        logging.debug(f"Train set length: {len(train_set)}, Validation set length: {len(validation_set)}, Test set length: {len(test_set)}")
    return train_sets, validation_sets, test_sets