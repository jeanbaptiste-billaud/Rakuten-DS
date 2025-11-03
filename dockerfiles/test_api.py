"""
Script de test pour l'API Rakuten Text Classification.
Usage: python test_api.py
"""

import requests
import json
from typing import Dict


API_URL = "http://localhost:8000"


def test_health():
    """Test du health check."""
    print("🏥 Test Health Check...")
    response = requests.get(f"{API_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   Services: {json.dumps(data['services'], indent=4)}")
    else:
        print(f"❌ Erreur: {response.status_code}")
    
    print()


def test_prediction(text: str):
    """Test de prédiction."""
    print(f"🔮 Test Prédiction...")
    print(f"   Texte: {text[:80]}...")
    
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Classe prédite: {data['predicted_class']}")
        print(f"   Confiance: {data['confidence']:.2%}")
        print(f"   Top 3 probabilités:")
        
        # Trier par probabilité décroissante
        sorted_probs = sorted(
            data['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for cls, prob in sorted_probs:
            print(f"      - Classe {cls}: {prob:.2%}")
    else:
        print(f"❌ Erreur: {response.status_code}")
        print(f"   Message: {response.text}")
    
    print()


def main():
    """Exécute les tests."""
    print("=" * 60)
    print("🧪 Test de l'API Rakuten Text Classification")
    print("=" * 60)
    print()
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Prédictions
    test_cases = [
        "Console de jeux vidéo PlayStation 5 avec manette sans fil DualSense",
        "Ordinateur portable Dell Inspiron 15 pouces, Intel Core i7, 16 GB RAM",
        "Robe femme élégante pour soirée, taille M, couleur noire",
        "Livre de cuisine française, recettes traditionnelles, 300 pages",
        "Smartphone Samsung Galaxy S23, 256 GB, écran AMOLED",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}")
        test_prediction(text)
    
    print("=" * 60)
    print("✅ Tests terminés")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Erreur: Impossible de se connecter à l'API")
        print("   Vérifiez que les services Docker sont démarrés:")
        print("   docker-compose up")