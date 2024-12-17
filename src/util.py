'''
A utility module that contains functions to save and extract data from pickles, jsons, and datasets.
'''
import datetime
import glob
import json
import logging
import os
import pickle
import torch
from typing import List, Tuple

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
    '''
    Extracts the tickers from the json files in the data directory.

    container: The container to store the data in.
    symbol: The symbol of the stock to extract the data from.
    '''
    files = glob.glob(os.path.join(DATA_PATH, symbol, "*.json"))
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    for f in sorted_files:
        extract_from_json(container, f)
    return container


def get_datasets():
    '''
    Returns a list of datasets from the pickles in the pickles directory.
    '''

    datasets = list(glob.glob(os.path.join(PICKLES_PATH, "*.pkl")))

    datasets: List[DataSet] = list([extract_from_pickle(dataset) for dataset in datasets])

    assert all(isinstance(dataset, DataSet) for dataset in datasets)
    return datasets


def get_train_validate_test_datasets(datasets: List[DataSet],
                                     timestamp1: datetime.datetime = datetime.datetime(2024, 1, 1),
                                     timestamp2: datetime.datetime = datetime.datetime(2024, 7, 1)) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Splits the datasets into train, validation, and test sets. The split is done based on the timestamps. see dataset.train_validation_test_split for more information.
    '''
    train_sets = []
    validation_sets = []
    test_sets = []

    for dataset in datasets:
        train_set, validation_set, test_set = dataset.train_validation_test_split(timestamp1, timestamp2)
        train_sets.append(train_set)
        validation_sets.append(validation_set)
        test_sets.append(test_set)
        logging.debug(
            f"Train set length: {len(train_set)}, Validation set length: {len(validation_set)}, Test set length: {len(test_set)}")
    return train_sets, validation_sets, test_sets
