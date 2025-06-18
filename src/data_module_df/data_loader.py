# data_loader.py
import os
import pandas as pd
import numpy as np

def load_training_data():
    X_path = '../../data/X_train_update.csv'
    y_path = '../../data/Y_train_CVw08PX.csv'

    X_df = pd.read_csv(X_path, index_col=0)
    y_df = pd.read_csv(y_path, index_col=0)
    return X_df, y_df

def combine_text_fields(X_df):
    return (X_df['designation'].fillna('') + ' ' + X_df['description'].fillna('')).values

def load_processed_npz(name, data_dir='../data/processed_data'):
    path = os.path.join(data_dir, f"{name}.npz")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)[f"{name}_"].item()
    else:
        raise FileNotFoundError(f"Fichier non trouvé : {path}")

def save_processed_npz(data, name, data_dir='../data/processed_data'):
    os.makedirs(data_dir, exist_ok=True)
    np.savez(os.path.join(data_dir, f"{name}.npz"), **{f"{name}_": data})

def load_index_split(path='../data/processed_data/indices_split.npz'):
    d = np.load(path)
    return {k: d[k] for k in d.files}

def save_index_split(indices_dict, path='../data/processed_data/indices_split.npz'):
    np.savez(path, **indices_dict)
