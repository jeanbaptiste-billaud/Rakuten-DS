from preprocess import ProductClassificationPipeline, PipelineConfig
from utils import generate_prediction_report, analyze_prediction_errors, plot_prediction_distribution
from data.processed_data.executer_pour_telecharger_donnees import telecharger_et_extraire_zip
import pandas as pd
import os
import yaml
import sys

url = "https://drive.google.com/file/d/1guhuHp0dVRPWCtZ7570jEsTub6m2RrRF/view?usp=sharing"
fichier_zip = "Preprocessed_data.zip"
dossier_donnees_pretraitees = "data/processed_data"
fichier_donnees_pretraitees = "data/processed_data/y_train.npz"


def load_model_configs():
    """
    Charge les configurations des modèles depuis le YAML
    
    Returns:
        dict: Configurations des modèles avec leurs paramètres
    """
    try:
        config_path = os.path.join('data', 'models', 'model_configs.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        
        # Vérification de la présence de tous les modèles attendus
        expected_models = {'xgboost', 'lightgbm', 'catboost', 'neural_net'}
        missing_models = expected_models - set(configs.keys())
        
        if missing_models:
            print(f"Attention : modèles manquants dans la configuration : {missing_models}")
            
        return configs
        
    except Exception as e:
        print(f"Erreur lors du chargement des configurations : {str(e)}")
        return {}

if __name__ == "__main__":
    # Forcer le prétraitement des données ou les télécharger si souhaité
    force_preprocess = False # Mettre à True pour forcer le preprocesing des données
    if force_preprocess == False and not os.path.exists(fichier_donnees_pretraitees):        
        # Appel de la fonction pour télécharger + extraire
        telecharger_et_extraire_zip(
            url=url,
            fichier_zip=fichier_zip,
            dossier_extraction=dossier_donnees_pretraitees
        )
        
    # Chargement des configurations
    config = PipelineConfig.from_yaml('config.yaml')
    pipeline = ProductClassificationPipeline(config)
    
    #Prétraitement
    try:
        pipeline.prepare_data(force_preprocess=force_preprocess)
    except Exception as e:
        print(f"Erreur lors du prétraitement : {str(e)}")
        sys.exit(1)
        
    # Chargement des configurations des modèles depuis le YAML
    models_to_test = load_model_configs()
    
    # Initialisation des dictionnaires pour stocker tous les résultats
    all_results = {}
    
    for model_name, params in models_to_test.items():
        print(f"\nTraitement  de {model_name}")
    
        # Vérification des prédictions existantes
        if pipeline.predictions_exist(model_name):
            predictions, probabilities = pipeline.load_predictions(model_name)
            pipeline.load_model(model_name)
            print(f"Prédictions chargées pour {model_name}")
        else:
            # Chargement ou entraînement du modèle
            model_dir = os.path.join(pipeline.config.model_path, model_name)
            if os.path.exists(model_dir):
                try:
                    pipeline.load_model(model_name)
                    print(f"Modèle {model_name} chargé")
                except Exception as e:
                    print(f"Erreur chargement {model_name}: {e}")
                    continue
            else:
                pipeline.train_model(model_type=model_name, **params)

            # Prévisions
            predictions, probabilities = pipeline.predict(pipeline.preprocessed_data['X_test_split'])
                        
            # Création du DataFrame de sortie
            pipeline.save_predictions(model_name, predictions, probabilities)

        file_path_rapport = os.path.join('data', 'rapports', f'rapport_{model_name}.csv')
        file_path_erreurs = os.path.join('data', 'erreurs', f'erreurs_{model_name}.csv')
        if os.path.exists(file_path_rapport):
            print(f"Le rapport {model_name} existe.")
        if os.path.exists(file_path_erreurs):
            print(f"Le fichier erreurs {model_name} existe.")
            
        evalute = True #Forcer l'évaluation (Maj des résultats)
        plot = False   #Pour ne pas afficher le graphe de distribution des prédictions déjà généré
        if not os.path.exists(file_path_erreurs) or not os.path.exists(file_path_rapport) or evalute==True:    
            # Prévisions et Évaluation
            all_results[model_name] = pipeline.evaluate()
            plot= True
            print(f"\nRésultats pour le modèle {model_name}:\n {all_results[model_name]}")

        # Création du DataFrame de résultats
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        
        # Création du dossier results s'il n'existe pas
        results_dir = os.path.join('data', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Sauvegarde des résultats en CSV
        results_path = os.path.join(results_dir, 'models_comparaison_results.csv')
        results_df.to_csv(results_path)
        
        # Génération des rapports pour chaque modèle
        if not os.path.exists(file_path_rapport):
            rapport = generate_prediction_report(pipeline, predictions, probabilities)
            rapport.to_csv(file_path_rapport, index=False)
            # Visualisation
            plot_prediction_distribution(rapport, model_name)
        if not os.path.exists(file_path_erreurs):
            erreurs = analyze_prediction_errors(
                pipeline,
                predictions,
                pipeline.preprocessed_data['y_test_split']
            )
            erreurs.to_csv(file_path_erreurs, index=False)
            
    if evalute:
        # Affichage des résultats
        print("\nRésultats de la comparaison:")
        print("-" * 50)
        for model_name, metrics in all_results.items():
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.4f}")