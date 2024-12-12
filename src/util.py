from dataset_container import DataSet, DataPoint
import glob, os, json, pickle

DATA_PATH = os.path.join('..', 'data')

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