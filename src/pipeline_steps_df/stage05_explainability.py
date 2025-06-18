# stage05_explainability.py
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

def run_explainability(pipeline, image_models_dict):
    if 'xgboost' not in image_models_dict:
        print("Explicabilité impossible sans modèle XGBoost")
        return

    pipeline.model = image_models_dict['xgboost']
    X = pipeline.preprocessed_data['X_test_split']['features']
    X_sample = X[:10]

    explainer = shap.TreeExplainer(pipeline.model)
    shap_values = explainer.shap_values(X_sample)

    os.makedirs('../../data/reports', exist_ok=True)
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig('../data/reports/shap_summary.png', bbox_inches='tight')
    plt.close()
