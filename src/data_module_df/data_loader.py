# data_loader.py
import os
import pandas as pd
import numpy as np

def load_training_data(data_dir):
    X_file = os.path.join(data_dir, 'X_train.csv')
    y_file = os.path.join(data_dir, 'Y_train.csv')

    X_df = pd.read_csv(X_file, index_col=0)
    y_df = pd.read_csv(y_file, index_col=0)
    return X_df, y_df

def combine_text_fields(X_df):
    return (X_df['designation'].fillna('') + ' ' + X_df['description'].fillna('')).values

def load_processed_npz(name, data_dir):
    path = os.path.join(data_dir, f"processed/{name}.npz")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)['arr_0'].item()
    else:
        raise FileNotFoundError(f"Fichier non trouvé : {path}")

def save_processed_npz(data, name, data_dir):
    os.makedirs(os.path.join(data_dir, "processed"), exist_ok=True)
    np.savez(os.path.join(data_dir, f"{name}.npz"), **{f"{name}_": data})

def load_index_split(data_dir):
    d = np.load(os.path.join(data_dir, "processed/index_split.npz"), allow_pickle=True)
    return {k: d[k] for k in d.files}

def save_index_split(indices_dict, data_dir):
    np.savez(os.path.join(data_dir, "processed/indices_split.npz"), **indices_dict)
