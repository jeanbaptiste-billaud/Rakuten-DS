# viz_utils.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Confusion Matrix', save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_prediction_distribution(proba_df, model_name, save_dir='../data/plots'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(proba_df['confidence'], bins=30, kde=True, color='blue')
    plt.title(f"Distribution des confiances - {model_name}")
    plt.xlabel("Confiance")
    plt.ylabel("Nombre de prédictions")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'distribution_confiance_{model_name}.png'))
    plt.close()


def analyze_prediction_errors(pipeline, y_pred, y_true):
    erreurs = []
    for i, (yp, yt) in enumerate(zip(y_pred, y_true)):
        if yp != yt:
            erreurs.append({
                'index': i,
                'predicted': yp,
                'true': yt
            })
    return pd.DataFrame(erreurs)
