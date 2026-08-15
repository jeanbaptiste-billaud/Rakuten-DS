import numpy as np
import numpy.testing as npt
import pytest
from sklearn.pipeline import Pipeline
# Importer la classe à tester
from src.models_module_df.model_text_classifier import TextClassifier


# --- Fixture 1: Données minimales ---

@pytest.fixture
def sample_data():
    """Un jeu de données minimal pour l'entraînement."""
    X_text = [
        "c'est un bon livre", 
        "un mauvais film", 
        "j'aime ce livre", 
        "je déteste ce film"
    ]
    y = np.array([1, 0, 1, 0]) # 1=Positif/Livre, 0=Négatif/Film
    return X_text, y

# --- Fixture 2: Classifieur entraîné ---

@pytest.fixture
def trained_classifier(sample_data):
    """
    Retourne une instance de TextClassifier DÉJÀ entraînée
    sur les données minimales.
    """
    X_text, y = sample_data
    # Le chemin n'a pas d'importance pour ce test, sauf pour save/load
    classifier = TextClassifier(model_path="dummy_path") 
    classifier.train(X_text, y)
    return classifier

# --- Tests des méthodes ---

def test_train_model_creation(trained_classifier):
    """
    Test 1: Vérifie que la méthode .train() crée bien
    un pipeline Scikit-learn avec les bons composants.
    """
    classifier = trained_classifier # La fixture fait le .train()
    
    assert classifier.model is not None
    assert isinstance(classifier.model, Pipeline)
    
    # Vérifier que les étapes du pipeline ont les bons noms
    assert 'tfidf' in classifier.model.named_steps
    assert 'svm' in classifier.model.named_steps
    
    # Vérifier qu'un des hyperparamètres est bien setté
    assert classifier.model.named_steps['svm'].C == 12
    assert classifier.model.named_steps['svm'].probability is True

def test_predict_output_format(trained_classifier):
    """
    Test 2: Vérifie que .predict() retourne des prédictions
    et des probabilités dans le bon format.
    """
    classifier = trained_classifier
    X_new = ["j'adore ce film", "un livre moyen"]
    
    preds, probs = classifier.predict(X_new)
    
    # Vérifier le type et la taille des prédictions
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 2
    
    # Vérifier le type et la taille des probabilités
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2, 2) # 2 échantillons, 2 classes (0 et 1)
    
    # Vérifier que les probabilités somment à 1 (avec une tolérance)
    npt.assert_allclose(probs.sum(axis=1), [1.0, 1.0])

def test_evaluate_logic():
    """
    Test 3: Teste la logique de la méthode .evaluate() 
    avec des données factices. C'est un test de logique pure.
    """
    classifier = TextClassifier(model_path="dummy")
    
    y_true = np.array([0, 1, 1, 0, 1])
    preds  = np.array([0, 1, 0, 0, 1]) # 4 corrects sur 5 -> Acc = 0.8
    probs  = np.array([
        [0.9, 0.1], # Conf 0.9
        [0.1, 0.9], # Conf 0.9
        [0.7, 0.3], # Conf 0.7 (erreur)
        [0.8, 0.2], # Conf 0.8
        [0.4, 0.6]  # Conf 0.6
    ])
    # Confiance moyenne = (0.9 + 0.9 + 0.7 + 0.8 + 0.6) / 5 = 3.9 / 5 = 0.78
    
    scores = classifier.evaluate(y_true, preds, probs)
    
    assert isinstance(scores, dict)
    assert scores['accuracy'] == 0.8000
    assert 'weighted_f1' in scores
    assert 'macro_f1' in scores
    
    # Vérifier le calcul de la confiance moyenne
    assert scores['mean_confidence'] == 0.7800

def test_save_and_load_roundtrip(trained_classifier, tmp_path):
    """
    Test 4: Teste un "aller-retour" (round-trip).
    1. Entraîne un modèle.
    2. Prédit sur un échantillon.
    3. Sauvegarde le modèle dans un dossier temporaire (tmp_path).
    4. Crée un NOUVEAU classifieur.
    5. Charge le modèle sauvegardé.
    6. Prédit à nouveau et vérifie que les résultats sont identiques.
    """
    # tmp_path est une fixture pytest qui fournit un dossier temporaire
    model_path = tmp_path / "test_model.pkl"
    
    # 1. Modèle 1 (original)
    classifier1 = trained_classifier
    classifier1.model_path = model_path # Mettre à jour le chemin
    
    # 2. Prédire
    X_test = ["un bon film ?"]
    preds1, probs1 = classifier1.predict(X_test)
    
    # 3. Sauvegarder
    classifier1.save()
    
    # Vérifier que le fichier et le dossier ont été créés
    assert model_path.parent.exists()
    assert model_path.exists()
    
    # 4. Modèle 2 (nouveau)
    classifier2 = TextClassifier(model_path=model_path)
    
    # 5. Charger
    classifier2.load()
    
    # 6. Prédire à nouveau
    assert classifier2.model is not None, "Le modèle n'a pas été chargé"
    preds2, probs2 = classifier2.predict(X_test)
    
    # 7. Vérifier l'identité
    npt.assert_array_equal(preds1, preds2)
    npt.assert_allclose(probs1, probs2)

def test_load_file_not_found():
    """
    Test 5: Vérifie que .load() lève une FileNotFoundError
    si le fichier n'existe pas.
    """
    classifier = TextClassifier(model_path="/un/chemin/qui/n/existe/pas/model.pkl")
    
    with pytest.raises(FileNotFoundError):
        classifier.load()
