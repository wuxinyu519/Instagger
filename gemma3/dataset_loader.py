import pickle

def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else [data]
