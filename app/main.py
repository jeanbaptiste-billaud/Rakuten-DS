##################################### 0. INITIALISATION COMMUNE (BASE SEULEMENT)
# Cette cellule contient uniquement l'initialisation de base

from preprocess import ProductClassificationPipeline, PipelineConfig
from utils import generate_prediction_report, analyze_prediction_errors, plot_prediction_distribution
from data.processed_data.executer_pour_telecharger_donnees import telecharger_et_extraire_zip
import pandas as pd
import numpy as np
import os
import sys
import yaml

# Configuration des URLs et chemins
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
        expected_models = {'xgboost', 'neural_net'}
        missing_models = expected_models - set(configs.keys())
        
        if missing_models:
            print(f"Attention : modèles manquants dans la configuration : {missing_models}")
            
        return configs
        
    except Exception as e:
        print(f"Erreur lors du chargement des configurations : {str(e)}")
        return {}

# Configuration du preprocessing
force_preprocess_image = False  # Mettre à True pour forcer le preprocessing des images
force_preprocess_texte = False  # Mettre à True pour forcer le preprocessing du texte

if not force_preprocess_image and not os.path.exists(fichier_donnees_pretraitees):        
    # Télécharger et extraire les données prétraitées
    telecharger_et_extraire_zip(
        url=url,
        fichier_zip=fichier_zip,
        dossier_extraction=dossier_donnees_pretraitees
    )

# Configuration et création du pipeline
print("1. Configuration du pipeline...")
config = PipelineConfig.from_yaml('config.yaml')
pipeline = ProductClassificationPipeline(config)

# Prétraitement des données (avec les bons paramètres)
print("2. Préparation des données de base...")
try:
    pipeline.prepare_data(
        force_preprocess_image=force_preprocess_image,
        force_preprocess_text=force_preprocess_texte
    )
    print("✓ Données de base préparées avec succès")
except Exception as e:
    print(f"Erreur lors du prétraitement : {str(e)}")
    sys.exit(1)

# Chargement des configurations des modèles
print("3. Chargement des configurations des modèles...")
models_config = load_model_configs()

# Sauvegarde des indices pour les tests futurs
print("4. Sauvegarde des indices...")
indices_dict = pipeline.extract_and_save_indices()

# Création des répertoires nécessaires
os.makedirs('data/reports', exist_ok=True)
os.makedirs('data/explanations', exist_ok=True)
os.makedirs('data/rapports', exist_ok=True)
os.makedirs('data/erreurs', exist_ok=True)
os.makedirs('data/results', exist_ok=True)

print("\n=== ✓ INITIALISATION COMMUNE TERMINÉE ===")
print(f"Pipeline configuré et données de base préparées")
print(f"Configurations des modèles chargées: {list(models_config.keys())}")
print(f"Indices sauvegardés pour {len(indices_dict)} ensembles de données")
print(f"Force preprocess image: {force_preprocess_image}")
print(f"Force preprocess texte: {force_preprocess_texte}")

##################################### 1. Gestion des modèles image (XGBoost et Neural Net)
# Vérification que l'initialisation a été faite
if 'pipeline' not in globals():
    print("ERREUR: Exécutez d'abord la cellule d'initialisation (Cellule 0)")
    raise RuntimeError("Pipeline non initialisé")

if __name__ == "__main__":
    # Vérification que l'initialisation a été faite
    if 'pipeline' not in globals():
        print("ERREUR: Exécutez d'abord la cellule d'initialisation (Cellule 0)")
        raise RuntimeError("Pipeline non initialisé")

    # Chargement des modèles IMAGE uniquement
    print("1. Chargement des modèles image...")
    image_models_dict = {}

    for model_name in ['xgboost', 'neural_net']:
        model_dir = os.path.join(pipeline.config.model_path, model_name)
        if os.path.exists(model_dir):
            try:
                pipeline.load_model(model_name)
                image_models_dict[model_name] = pipeline.model
                print(f"✓ Modèle {model_name} chargé")
            except Exception as e:
                print(f"✗ Erreur chargement {model_name}: {e}")
        else:
            print(f"✗ Modèle {model_name} non trouvé dans {model_dir}")

    if not image_models_dict:
        print("ERREUR: Aucun modèle image disponible")
        raise RuntimeError("Aucun modèle image chargé")

    # Initialisation des résultats
    all_results = {}
    models_to_test = {k: v for k, v in models_config.items() if k in image_models_dict.keys()}

    print(f"2. Modèles image à tester: {list(models_to_test.keys())}")
    
    for model_name, params in models_to_test.items():
        print(f"\nTraitement  de {model_name}")
        # Recherche d'hyperparamètres (optionnel)
        recherche_hyperparametres = False
        if recherche_hyperparametres:
            print(f"\nRecherche des hyperparamètres pour {model_name}")            
            # Optimisation des hyperparamètres
            best_params = pipeline.optimize_hyperparameters(model_name)
            pipeline.save_hyperparameters(model_name, best_params)
            params.update(best_params)
    
        # Charger le modèle dans le pipeline (Nouveau, à tester, remplace "pipeline.load_model(model_name)")
        pipeline.model = image_models_dict[model_name]
        
        # Vérification et chargement des prédictions
        if pipeline.predictions_exist(model_name):
            predictions, probabilities = pipeline.load_predictions(model_name)
            print(f"✓ Prédictions chargées pour {model_name}")
        else:
            print(f"Génération des prédictions pour {model_name}...")
            # Générer les prédictions sur les données image
            predictions, probabilities = pipeline.predict(pipeline.preprocessed_data['X_test_split'])
            
            # Sauvegarder les prédictions
            pipeline.save_predictions(model_name, predictions, probabilities)
            print(f"✓ Prédictions générées et sauvegardées pour {model_name}")

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
        plot_distribution  = False   #Pour ne pas afficher le graphe de distribution des prédictions déjà généré
        if not os.path.exists(file_path_erreurs) or not os.path.exists(file_path_rapport) or evalute==True:    
            print(f"Évaluation du modèle image {model_name}...")
            # Prévisions et Évaluation
            all_results[model_name] = pipeline.evaluate()
            plot_distribution = True
            
            print(f"✓ Résultats pour {model_name}:")
            # Affichage des métriques principales
            main_metrics = ['accuracy', 'weighted_f1', 'mean_confidence']
            for metric in main_metrics:
                if metric in all_results[model_name]:
                    print(f"  {metric}: {all_results[model_name][metric]:.4f}")

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
            if plot_distribution:
                plot_prediction_distribution(rapport, model_name)
        if not os.path.exists(file_path_erreurs):
            erreurs = analyze_prediction_errors(
                pipeline,
                predictions,
                pipeline.preprocessed_data['y_test_split']
            )
            erreurs.to_csv(file_path_erreurs, index=False)
            
    if all_results:
        # Création du DataFrame de résultats
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        
        # Sauvegarde des résultats en CSV
        results_path = os.path.join('data', 'results', 'image_models_comparison_results.csv')
        results_df.to_csv(results_path)
        print(f"✓ Résultats de comparaison IMAGE sauvegardés: {results_path}")

        # Affichage du résumé des résultats IMAGE
        print("\n=== RÉSUMÉ DES PERFORMANCES IMAGE ===")
        for model_name, metrics in all_results.items():
            print(f"\n{model_name.upper()}:")
            main_metrics = ['accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall', 'mean_confidence']
            for metric_name in main_metrics:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, (int, float)):
                        print(f"  {metric_name}: {value:.4f}")

    print(f"\n=== ✓ ÉVALUATION MODÈLES IMAGE TERMINÉE ===")
    print(f"Modèles image évalués: {list(all_results.keys()) if all_results else 'Aucun (tous les rapports existaient déjà)'}")
    print(f"Les modèles image sont prêts pour l'analyse multimodale")

    # Rendre les modèles disponibles pour la cellule suivante
    print(f"Modèles image disponibles pour l'analyse multimodale: {list(image_models_dict.keys())}")
    
##################################### 2. Gestion du modèle texte SVM
# ÉVALUATION DU MODÈLE TEXTE UNIQUEMENT
# Cette cellule traite uniquement le modèle de texte (SVM) en utilisant le pipeline

# Vérification que l'initialisation a été faite
if 'pipeline' not in globals():
    print("ERREUR: Exécutez d'abord la cellule d'initialisation (Cellule 0)")
    raise RuntimeError("Pipeline non initialisé")

print("1. Chargement du modèle texte...")
text_model_loaded = pipeline.load_text_model('SVM')

print("2. Préparation des données texte...")
X_text_test, y_text_test = pipeline.prepare_text_data()

# Vérification des prédictions existantes
model_name_text = 'SVM'
print("3. Vérification des prédictions existantes...")

if pipeline.text_predictions_exist(model_name_text):
    print(f"✓ Chargement des prédictions existantes pour {model_name_text}")
    predictions_text, probabilities_text = pipeline.load_text_predictions(model_name_text)
else:
    print(f"Génération des prédictions pour {model_name_text}...")
    predictions_text, probabilities_text = pipeline.predict_text()
    pipeline.save_text_predictions(predictions_text, probabilities_text, model_name_text)

print("4. Évaluation du modèle texte...")
results_text, _, _ = pipeline.evaluate_text_model()

# Chemins des fichiers de rapport pour le texte
file_path_rapport_text = os.path.join('data', 'rapports', f'rapport_text_{model_name_text}.csv')
file_path_erreurs_text = os.path.join('data', 'erreurs', f'erreurs_text_{model_name_text}.csv')

rapport_text_exists = os.path.exists(file_path_rapport_text)
erreurs_text_exists = os.path.exists(file_path_erreurs_text)

if rapport_text_exists:
    print(f"✓ Le rapport texte {model_name_text} existe.")
if erreurs_text_exists:
    print(f"✓ Le fichier erreurs texte {model_name_text} existe.")

# Génération des rapports spécifiques au texte si nécessaire
evaluer_text = True  # Mettre à False pour éviter la réévaluation

if not rapport_text_exists or evaluer_text:
    print(f"6. Génération du rapport détaillé pour le modèle texte...")
    
    # Créer un rapport spécifique au texte
    rapport_text_data = []
    
    for i, (pred, true_label) in enumerate(zip(predictions_text, y_text_test)):
        prob_max = np.max(probabilities_text[i])
        pred_class_name = pipeline.category_names.get(pred, f"Unknown_{pred}")
        true_class_name = pipeline.category_names.get(true_label, f"Unknown_{true_label}")
        
        rapport_text_data.append({
            'sample_id': i,
            'true_label': true_label,
            'true_class_name': true_class_name,
            'predicted_label': pred,
            'predicted_class_name': pred_class_name,
            'confidence': prob_max,
            'correct': pred == true_label,
            'text_sample': X_text_test[i][:100] + "..." if len(X_text_test[i]) > 100 else X_text_test[i]
        })
    
    rapport_text_df = pd.DataFrame(rapport_text_data)
    rapport_text_df.to_csv(file_path_rapport_text, index=False)
    print(f"✓ Rapport texte sauvegardé: {file_path_rapport_text}")

if not erreurs_text_exists or evaluer_text:
    print(f"7. Analyse des erreurs pour le modèle texte...")
    
    # Analyse des erreurs spécifique au texte
    erreurs_text_data = []
    
    for i, (pred, true_label) in enumerate(zip(predictions_text, y_text_test)):
        if pred != true_label:  # Seulement les erreurs
            prob_max = np.max(probabilities_text[i])
            pred_class_name = pipeline.category_names.get(pred, f"Unknown_{pred}")
            true_class_name = pipeline.category_names.get(true_label, f"Unknown_{true_label}")
            
            erreurs_text_data.append({
                'sample_id': i,
                'true_label': true_label,
                'true_class_name': true_class_name,
                'predicted_label': pred,
                'predicted_class_name': pred_class_name,
                'confidence': prob_max,
                'error_type': 'low_confidence' if prob_max < 0.5 else 'high_confidence',
                'text_sample': X_text_test[i][:200] + "..." if len(X_text_test[i]) > 200 else X_text_test[i]
            })
    
    erreurs_text_df = pd.DataFrame(erreurs_text_data)
    erreurs_text_df.to_csv(file_path_erreurs_text, index=False)
    print(f"✓ Analyse des erreurs texte sauvegardée: {file_path_erreurs_text}")

# Sauvegarde des résultats du modèle texte
results_text_path = os.path.join('data', 'results', 'text_model_results.csv')
results_text_df = pd.DataFrame([results_text], index=[model_name_text])
results_text_df.to_csv(results_text_path)

print("\n=== RÉSUMÉ DES PERFORMANCES TEXTE ===")
print(f"Modèle: {model_name_text}")
print(f"Échantillons: {results_text['num_samples']}")
print(f"Accuracy: {results_text['accuracy']:.4f}")
print(f"F1-score pondéré: {results_text['weighted_f1']:.4f}")
print(f"F1-score macro: {results_text['macro_f1']:.4f}")
print(f"Confiance moyenne: {results_text['mean_confidence']:.4f}")
print(f"Échantillons faible confiance (<0.5): {results_text['low_confidence_samples']}")
print(f"Échantillons haute confiance (>0.8): {results_text['high_confidence_samples']}")

print(f"\n=== ✓ ÉVALUATION MODÈLE TEXTE TERMINÉE ===")
print(f"Le modèle texte {model_name_text} est prêt pour l'analyse multimodale")
print(f"Résultats sauvegardés dans: {results_text_path}")

# Rendre le modèle texte disponible pour l'analyse multimodale
text_model_ready = True
text_model_name = model_name_text
print(f"Variables globales créées: text_model_ready, text_model_name")

##################################### 3. Fusion des modèles texte et image
# ====================  FUSION MULTIMODALE ====================
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score

print("=== ANALYSE MULTIMODALE ===")

# Vérification des prérequis
if 'pipeline' not in globals():
    print("ERREUR: Exécuter d'abord la cellule d'initialisation (Cellule 0)")
    raise RuntimeError("Pipeline non initialisé")

print("1. Configuration des modèles multimodaux...")
# Charger tous les modèles nécessaires
models_to_test = {}

# Modèles image
for model_name in ['xgboost', 'neural_net']:
    model_dir = os.path.join(pipeline.config.model_path, model_name)
    if os.path.exists(model_dir):
        try:
            pipeline.load_model(model_name)
            models_to_test[model_name] = pipeline.model
            print(f"✓ Modèle image {model_name} chargé")
        except Exception as e:
            print(f"✗ Erreur chargement {model_name}: {e}")

# Modèle texte
print("Chargement du modèle texte...")
try:
    text_model_loaded = pipeline.load_text_model('SVM')
    if text_model_loaded:
        models_to_test['text'] = pipeline.text_model
        print(f"✓ Modèle texte SVM chargé")
    else:
        print(f"⚠ Modèle texte à générer (Exécuter Cellule 2)")
except Exception as e:
    print(f"✗ Erreur chargement modèle texte: {e}")
    raise

if len(models_to_test) < 2:
    print("ERREUR: Pas assez de modèles disponibles pour l'analyse multimodale")
    raise RuntimeError("Modèles insuffisants")

print("2. Préparation des données multimodales...")
# Préparation des données texte - autonome
print("Préparation des données texte...")
try:
    X_text_test, y_text_test = pipeline.prepare_text_data()
    print(f"✓ Données texte préparées: {len(X_text_test)} échantillons")
except Exception as e:
    print(f"✗ Erreur préparation données texte: {e}")
    raise

# Données image
X_image_test = pipeline.preprocessed_data['X_test_split']
if isinstance(X_image_test, dict):
    X_image_test = X_image_test['features']
elif X_image_test.shape == ():
    X_image_test = X_image_test.item()
    if isinstance(X_image_test, dict) and 'features' in X_image_test:
        X_image_test = X_image_test['features']

print(f"✓ Données image extraites: {X_image_test.shape}")

# Alignement des dimensions
min_size = min(len(X_text_test), len(X_image_test), len(y_text_test))
X_text_test = X_text_test[:min_size]
X_image_test = X_image_test[:min_size]
y_test = y_text_test[:min_size]

print(f"Dimensions alignées: {min_size} échantillons")

print("3. Évaluation des combinaisons multimodales...")
# Dictionnaire pour stocker tous les résultats
all_results = {}
fusion_strategies = ['mean', 'product', 'weighted', 'confidence_weighted']

# Évaluation du texte seul
if 'text' in models_to_test:
    print("Évaluation du modèle texte seul...")
    text_predictions, text_probabilities = pipeline.predict_text(X_text_test)
    
    # OPTIMISATION: Stocker pour réutilisation
    cached_text_predictions = text_predictions
    cached_text_probabilities = text_probabilities
    
    text_accuracy = accuracy_score(y_test, text_predictions)
    text_f1 = f1_score(y_test, text_predictions, average='weighted')
    
    all_results['text_only'] = {
        'accuracy': text_accuracy,
        'weighted_f1': text_f1,
        'model_type': 'text'
    }
    print(f"Texte seul: accuracy={text_accuracy:.4f}, f1={text_f1:.4f}")

# Évaluation des modèles image seuls et combinaisons multimodales
for model_name in ['xgboost', 'neural_net']:
    if model_name not in models_to_test:
        continue
        
    print(f"\nÉvaluation du modèle {model_name}...")
    
    # Charger le modèle dans le pipeline
    pipeline.model = models_to_test[model_name]
    
    # Prédictions image seules
    try:
        image_predictions, image_probabilities = pipeline.predict(X_image_test)
        
        # Convertir en indices si nécessaire pour la comparaison
        if isinstance(image_predictions[0], (int, np.integer)) and max(image_predictions) < 100:
            image_pred_indices = image_predictions
        else:
            image_pred_indices = np.array([pipeline.category_to_idx.get(code, 0) for code in image_predictions])
        
        y_test_indices = np.array([pipeline.category_to_idx.get(code, 0) for code in y_test])
        
        image_accuracy = accuracy_score(y_test_indices, image_pred_indices)
        image_f1 = f1_score(y_test_indices, image_pred_indices, average='weighted')
        
        all_results[f'{model_name}_only'] = {
            'accuracy': image_accuracy,
            'weighted_f1': image_f1,
            'model_type': 'image'
        }
        print(f"{model_name} seul: accuracy={image_accuracy:.4f}, f1={image_f1:.4f}")
        
        # Combinaisons multimodales
        for strategy in fusion_strategies:
            print(f"  Fusion {strategy}...")
            
            # S'assurer que les probabilités ont les mêmes dimensions
            if text_probabilities.shape[1] != image_probabilities.shape[1]:
                print(f"    ATTENTION: Dimensions différentes (texte: {text_probabilities.shape}, image: {image_probabilities.shape})")
                raise RuntimeError("Problème d'alignement entre les dimensions des probabilités texte et image")
            else:
                text_probs_adj = text_probabilities
                image_probs_adj = image_probabilities
            
            # Fusion des probabilités
            fused_probabilities = pipeline.fuse_predictions(text_probs_adj, image_probs_adj, strategy)
            fused_predictions = np.argmax(fused_probabilities, axis=1)
            
            # Évaluation
            fused_accuracy = accuracy_score(y_test_indices, fused_predictions)
            fused_f1 = f1_score(y_test_indices, fused_predictions, average='weighted')
            
            all_results[f'{model_name}_{strategy}'] = {
                'accuracy': fused_accuracy,
                'weighted_f1': fused_f1,
                'model_type': 'multimodal',
                'fusion_strategy': strategy
            }
            print(f"    accuracy={fused_accuracy:.4f}, f1={fused_f1:.4f}")
            
    except Exception as e:
        print(f"Erreur avec le modèle {model_name}: {e}")
        all_results[f'{model_name}_error'] = {'error': str(e)}

print("4. Sauvegarde des résultats...")
# Conversion en DataFrame et sauvegarde
results_df = pd.DataFrame.from_dict(all_results, orient='index')
results_path = os.path.join('data', 'reports', 'multimodal_comparison_results.csv')
os.makedirs('data/reports', exist_ok=True)
results_df.to_csv(results_path)

print("5. Identification de la meilleure combinaison...")
# Filtrer les résultats valides (sans erreurs)
valid_results = {k: v for k, v in all_results.items() if 'error' not in v}

if valid_results:
    # Trouver la meilleure combinaison
    best_combo = max(valid_results.keys(), key=lambda x: valid_results[x]['weighted_f1'])
    best_f1 = valid_results[best_combo]['weighted_f1']
    
    print(f"\n=== RÉSULTATS MULTIMODAUX ===")
    print(f"Meilleure combinaison: {best_combo}")
    print(f"F1-score: {best_f1:.4f}")
    print(f"Accuracy: {valid_results[best_combo]['accuracy']:.4f}")
    
    print(f"\nTop 5 des combinaisons:")
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['weighted_f1'], reverse=True)
    for i, (combo, metrics) in enumerate(sorted_results[:5], 1):
        print(f"{i}. {combo}: F1={metrics['weighted_f1']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Rapport détaillé pour la meilleure combinaison
    if 'multimodal' in valid_results[best_combo].get('model_type', ''):
        print(f"\n6. Génération du rapport détaillé pour {best_combo}...")
        
        # Reconstituer les prédictions pour la meilleure combinaison
        model_name = best_combo.split('_')[0]
        strategy = '_'.join(best_combo.split('_')[1:])
        
        # Recharger le modèle approprié
        pipeline.model = models_to_test[model_name]
        
        # Refaire les prédictions
        text_preds, text_probs = pipeline.predict_text(X_text_test)
        image_preds, image_probs = pipeline.predict(X_image_test)
        
        # Fusion
        fused_probs = pipeline.fuse_predictions(text_probs, image_probs, strategy)
        fused_preds = np.argmax(fused_probs, axis=1)
        
        # Créer le rapport multimodal
        rapport_multimodal = pipeline.create_combined_prediction_report(
            text_predictions=text_preds,
            text_probabilities=text_probs,
            image_predictions=image_preds,
            image_probabilities=image_probs,
            fused_predictions=fused_preds,
            fused_probabilities=fused_probs,
            y_true=y_test,
            fusion_strategy=strategy
        )
        
        # Sauvegarder le rapport
        rapport_path = os.path.join('data', 'reports', f'rapport_multimodal_{best_combo}.csv')
        rapport_multimodal.to_csv(rapport_path, index=False)
        print(f"✓ Rapport multimodal sauvegardé: {rapport_path}")
        
        # Statistiques du rapport
        total_samples = len(rapport_multimodal)
        fusion_improved = rapport_multimodal['fusion_improved'].sum()
        agreement_rate = rapport_multimodal['agreement_text_image'].mean()
        
        print(f"\nStatistiques du rapport:")
        print(f"- Échantillons total: {total_samples}")
        print(f"- Fusion améliorée: {fusion_improved} ({fusion_improved/total_samples*100:.1f}%)")
        print(f"- Accord texte-image: {agreement_rate*100:.1f}%")

else:
    print("ERREUR: Aucun résultat valide obtenu")

print(f"\n=== ✓ ANALYSE MULTIMODALE TERMINÉE ===")
print(f"Résultats sauvegardés dans: {results_path}")
print(f"Modèles évalués: {len(valid_results)} combinaisons")


##################################### 4. Explicabilité des modèles : SHAP
# ==================== SHAP ULTRA-SIMPLE AVEC GRAPHIQUES NATIFS ====================
import numpy as np
import matplotlib.pyplot as plt
import shap
import traceback

print("=== SHAP SIMPLE AVEC GRAPHIQUES NATIFS ===")

try:
    # ==================== 1. PRÉPARATION ULTRA-SIMPLE ====================
    print("1. Préparation des données...")
    
    # Prendre un très petit échantillon
    n_samples = 10
    indices = np.random.choice(len(X_image_test), min(n_samples, len(X_image_test)), replace=False)
    X_sample = X_image_test[indices]
    
    print(f"   Échantillon: {X_sample.shape}")
    print(f"   Model XGBoost disponible: {'xgboost' in models_to_test}")
    
    # ==================== 2. EXPLICATIONS SHAP NATIVES ====================
    print("2. Génération des explications SHAP...")
    
    # Utiliser TreeExplainer (le plus simple et rapide)
    model = models_to_test['xgboost']
    explainer = shap.TreeExplainer(model)
    
    print("   TreeExplainer créé avec succès")
    
    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(X_sample)
    print(f"   SHAP values calculées: {type(shap_values)}")
    
    # Gérer le format des valeurs SHAP
    if isinstance(shap_values, list):
        print(f"   Liste de {len(shap_values)} éléments")
        shap_matrix = shap_values[0] if len(shap_values) > 0 else shap_values[0]
    elif isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 3:
            print(f"   Format 3D: {shap_values.shape} -> prendre première classe")
            shap_matrix = shap_values[:, :, 0]
        else:
            shap_matrix = shap_values
    
    print(f"   SHAP matrix finale: {shap_matrix.shape}")
    
    # ==================== 3. GRAPHIQUES SHAP NATIFS ====================
    print("3. Génération des graphiques SHAP...")
    
    # Créer feature names simples
    feature_names = [f'Feature_{i}' for i in range(X_sample.shape[1])]
    
    # GRAPHIQUE 1: Summary Plot (Bar)
    print("   Graphique 1: Summary plot (bar)")
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, 
                         plot_type='bar', max_display=20, show=False)
    else:
        if len(shap_values.shape) == 3:
            shap.summary_plot(shap_values[:, :, 0], X_sample, feature_names=feature_names, 
                             plot_type='bar', max_display=20, show=False)
        else:
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                             plot_type='bar', max_display=20, show=False)
    
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.savefig("data/reports/shap_bar_plot.png", dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # GRAPHIQUE 2: Summary Plot (Dot)
    print("   Graphique 2: Summary plot (dot)")
    plt.figure(figsize=(10, 6))
    try:
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, 
                             max_display=15, show=False)
        else:
            if len(shap_values.shape) == 3:
                shap.summary_plot(shap_values[:, :, 0], X_sample, feature_names=feature_names, 
                                 max_display=15, show=False)
            else:
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                                 max_display=15, show=False)
        
        plt.title("SHAP Feature Impact (Dot Plot)")
        plt.tight_layout()
        plt.savefig("data/reports/shap_dot_plot.png", dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"   Erreur dot plot: {e}")
        plt.close()
    
    # GRAPHIQUE 3: Waterfall pour le premier exemple
    print("   Graphique 3: Waterfall plot")
    try:
        plt.figure(figsize=(10, 8))
        
        if isinstance(shap_values, list):
            shap_vals_example = shap_values[0][0]
            expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            if len(shap_values.shape) == 3:
                shap_vals_example = shap_values[0, :, 0]
                expected_value = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                shap_vals_example = shap_values[0]
                expected_value = explainer.expected_value
        
        # Créer un objet Explanation pour waterfall
        explanation = shap.Explanation(
            values=shap_vals_example,
            base_values=expected_value,
            data=X_sample[0],
            feature_names=feature_names
        )
        
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title("SHAP Waterfall - Premier Exemple")
        plt.tight_layout()
        plt.savefig("data/reports/shap_waterfall.png", dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"   Erreur waterfall: {e}")
        plt.close()
    
    # GRAPHIQUE 4: Force Plot (si possible)
    print("   Graphique 4: Force plot")
    try:
        if isinstance(shap_values, list):
            shap_vals_force = shap_values[0][0]
            expected_value_force = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            if len(shap_values.shape) == 3:
                shap_vals_force = shap_values[0, :, 0]
                expected_value_force = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                shap_vals_force = shap_values[0]
                expected_value_force = explainer.expected_value
        
        # Force plot en matplotlib
        shap.force_plot(
            expected_value_force, 
            shap_vals_force, 
            X_sample[0], 
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title("SHAP Force Plot - Premier Exemple")
        plt.tight_layout()
        plt.savefig("data/reports/shap_force_plot.png", dpi=200, bbox_inches='tight')
        plt.show()
        plt.close()
        
    except Exception as e:
        print(f"   Force plot non disponible: {e}")
    
    # ==================== 4. RÉSUMÉ SIMPLE ====================
    print("4. Résumé des résultats...")
    
    # Calculer importance moyenne
    mean_importance = np.mean(np.abs(shap_matrix), axis=0)
    top_5_indices = np.argsort(mean_importance)[-5:]
    
    print(f"\n📊 TOP 5 FEATURES LES PLUS IMPORTANTES:")
    for i, idx in enumerate(reversed(top_5_indices), 1):
        print(f"   {i}. Feature_{idx}: {mean_importance[idx]:.4f}")
    
    print(f"\n📈 STATISTIQUES SHAP:")
    print(f"   Échantillons analysés: {shap_matrix.shape[0]}")
    print(f"   Features totales: {shap_matrix.shape[1]}")
    print(f"   Impact moyen: {np.mean(shap_matrix):.4f}")
    print(f"   Impact max: {np.max(shap_matrix):.4f}")
    print(f"   Impact min: {np.min(shap_matrix):.4f}")
    
    # ==================== 5. COMPARAISON PERFORMANCES ====================
    print("\n5. Rappel des performances...")
    
    if 'valid_results' in globals():
        print(f"\n🏆 CLASSEMENT DES MODÈLES:")
        
        # Trier par F1-score
        sorted_results = sorted(valid_results.items(), 
                               key=lambda x: x[1].get('weighted_f1', 0), 
                               reverse=True)
        
        for i, (model_name, metrics) in enumerate(sorted_results[:5], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            f1_score = metrics.get('weighted_f1', 0)
            accuracy = metrics.get('accuracy', 0)
            print(f"   {medal} {i}. {model_name}: F1={f1_score:.3f}, Acc={accuracy:.3f}")
    
    print("\n✅ ANALYSE SHAP TERMINÉE!")
    print("📁 Graphiques générés:")
    print("   - data/reports/shap_bar_plot.png")
    print("   - data/reports/shap_dot_plot.png") 
    print("   - data/reports/shap_waterfall.png")
    print("   - data/reports/shap_force_plot.png (si disponible)")
    
except Exception as e:
    print(f"❌ Erreur générale: {e}")
    print(f"💡 Détails: {traceback.format_exc()}")
    
    # Fallback ultra-simple
    print("\n🔄 Tentative de fallback ultra-simple...")
    try:
        # Juste afficher l'importance native du modèle
        if 'models_to_test' in globals() and 'xgboost' in models_to_test:
            model = models_to_test['xgboost']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                plt.figure(figsize=(10, 6))
                top_n = min(20, len(importances))
                top_indices = np.argsort(importances)[-top_n:]
                
                plt.barh(range(top_n), importances[top_indices])
                plt.yticks(range(top_n), [f'Feature_{i}' for i in top_indices])
                plt.xlabel('Importance')
                plt.title('XGBoost Native Feature Importance')
                plt.tight_layout()
                plt.savefig("data/reports/fallback_importance.png", dpi=200, bbox_inches='tight')
                plt.show()
                plt.close()
                
                print("✅ Graphique de fallback généré: fallback_importance.png")
            else:
                print("❌ Pas de feature_importances_ disponible")
        else:
            print("❌ Modèle XGBoost non trouvé")
    except Exception as e2:
        print(f"❌ Fallback échoué aussi: {e2}")