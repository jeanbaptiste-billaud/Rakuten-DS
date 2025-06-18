# data_balancing.py
import numpy as np
import pandas as pd
from collections import Counter


def undersample(X, y, max_per_class=1000, random_state=42):
    np.random.seed(random_state)
    df = pd.DataFrame({'X': list(X), 'y': y})
    sampled_dfs = []

    for label in np.unique(y):
        df_class = df[df['y'] == label]
        if len(df_class) > max_per_class:
            sampled = df_class.sample(n=max_per_class, random_state=random_state)
        else:
            sampled = df_class
        sampled_dfs.append(sampled)

    balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return np.array(balanced_df['X'].tolist()), balanced_df['y'].values


def print_class_distribution(y, title="Répartition des classes"):
    counter = Counter(y)
    print(f"\n📊 {title}:")
    for label, count in counter.items():
        print(f"  Classe {label}: {count} échantillons")
