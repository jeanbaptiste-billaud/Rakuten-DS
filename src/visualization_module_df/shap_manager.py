# shap_manager.py
import shap
import matplotlib.pyplot as plt
import os


def explain_xgboost_model(model, X_sample, feature_names=None, save_path='../data/reports/shap_summary.png'):
    """
    Génère un graphique SHAP de résumé pour un modèle XGBoost.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def explain_prediction(model, X_instance, feature_names=None):
    """
    Affiche une explication locale pour une instance.
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X_instance)
    shap.plots.waterfall(shap_values[0], feature_names=feature_names)
