# model_utils.py
import os
import yaml
import numpy as np
from sklearn.model_selection import cross_val_score


def load_model_configs():
    """
    Charge les configurations des modèles depuis un fichier YAML
    Returns:
        dict: Configurations des modèles avec leurs paramètres
    """
    try:
        config_path = os.path.join('../../data', 'models', 'train_image.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)

        expected_models = {'xgboost', 'neural_net'}
        missing_models = expected_models - set(configs.keys())
        if missing_models:
            print(f"⚠️ Modèles manquants dans la configuration : {missing_models}")

        return configs

    except Exception as e:
        print(f"Erreur lors du chargement des configurations : {str(e)}")
        return {}


def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    """
    Effectue une validation croisée sur un modèle sklearn-like
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'all_scores': scores
    }


def optimize_hyperparameters(objective_fn, n_trials=50, direction='maximize'):
    """
    Optimise les hyperparamètres d'un modèle à l'aide d'Optuna.

    Args:
        objective_fn (callable): fonction objectif à optimiser
        n_trials (int): nombre d'essais
        direction (str): 'minimize' ou 'maximize'

    Returns:
        dict: meilleurs hyperparamètres trouvés
    """
    import optuna
    study = optuna.create_study(direction=direction)
    study.optimize(objective_fn, n_trials=n_trials)
    print(f"✅ Meilleur score : {study.best_value}")
    print(f"🏆 Meilleurs paramètres : {study.best_params}")
    return study.best_params
