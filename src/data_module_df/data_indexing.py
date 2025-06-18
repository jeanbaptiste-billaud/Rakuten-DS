# data_indexing.py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os


def generate_splits(y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Génère les indices pour train, validation et test de manière stratifiée
    """
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(sss1.split(np.zeros(len(y)), y))

    y_train_val = y[train_val_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(sss2.split(np.zeros(len(y_train_val)), y_train_val))

    indices = {
        'train_idx': train_val_idx[train_idx],
        'val_idx': train_val_idx[val_idx],
        'test_idx': test_idx
    }
    return indices


def save_indices(indices_dict, path='../data/processed_data/indices_split.npz'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **indices_dict)


def load_indices(path='../data/processed_data/indices_split.npz'):
    return {k: v for k, v in np.load(path).items()}
