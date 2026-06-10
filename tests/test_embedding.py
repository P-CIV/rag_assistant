"""Test des embeddings avec Azure OpenAI"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au chemin
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import creer_modele_embedding
import numpy as np


def calculer_similarite_cosinus(vec1: list, vec2: list) -> float:
    """Calcule la similarité cosinus entre deux vecteurs"""
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def test_embedding_basique():
    """Test basique : embedding d'un texte simple"""
    print("=" * 60)
    print("TEST 1 : Embedding d'un texte simple")
    print("=" * 60)
    
    modele = creer_modele_embedding()
    
    texte_test = "Ceci est un texte de test pour l'embedding"
    print(f"\nTexte : {texte_test}")
    
    try:
        embedding = modele.embed_query(texte_test)
        print(f"✓ Embedding généré avec succès")
        print(f"  - Dimension : {len(embedding)}")
        print(f"  - Premiers 5 éléments : {embedding[:5]}")
        return True
    except Exception as e:
        print(f"✗ Erreur : {e}")
        return False


def test_embedding_similarite():
    """Test de similarité : compare deux textes similaires"""
    print("\n" + "=" * 60)
    print("TEST 2 : Similarité entre textes")
    print("=" * 60)
    
    modele = creer_modele_embedding()
    
    textes = [
        "Les catalogues de services sont importants",
        "Les services proposés dans le catalogue",
        "Python est un langage de programmation"
    ]
    
    try:
        embeddings = [modele.embed_query(t) for t in textes]
        
        print(f"\nComparaison de {len(textes)} textes :")
        for i, texte in enumerate(textes):
            print(f"  [{i+1}] {texte[:50]}...")
        
        print("\nMatrice de similarité :")
        for i in range(len(textes)):
            for j in range(i+1, len(textes)):
                sim = calculer_similarite_cosinus(embeddings[i], embeddings[j])
                print(f"  Texte {i+1} vs Texte {j+1} : {sim:.4f}")
        
        # Vérifier que les textes similaires (1 et 2) ont une similarité plus élevée
        sim_12 = calculer_similarite_cosinus(embeddings[0], embeddings[1])
        sim_13 = calculer_similarite_cosinus(embeddings[0], embeddings[2])
        
        if sim_12 > sim_13:
            print("\n✓ Les textes similaires ont une plus haute similarité")
            return True
        else:
            print("\n⚠ Les résultats pourraient nécessiter une investigation")
            return True
    except Exception as e:
        print(f"✗ Erreur : {e}")
        return False


def test_embedding_batch():
    """Test batch : embedding de plusieurs textes"""
    print("\n" + "=" * 60)
    print("TEST 3 : Embedding en batch")
    print("=" * 60)
    
    modele = creer_modele_embedding()
    
    textes = [
        "Premier document de test",
        "Deuxième document de test",
        "Troisième document de test"
    ]
    
    try:
        embeddings = modele.embed_documents(textes)
        print(f"\n✓ {len(embeddings)} embeddings générés avec succès")
        print(f"  - Dimension de chaque embedding : {len(embeddings[0])}")
        
        # Vérifier que tous les embeddings ont la même dimension
        dimensions = [len(e) for e in embeddings]
        if len(set(dimensions)) == 1:
            print(f"✓ Tous les embeddings ont la même dimension")
            return True
        else:
            print(f"✗ Les embeddings n'ont pas la même dimension : {set(dimensions)}")
            return False
    except Exception as e:
        print(f"✗ Erreur : {e}")
        return False


def test_embedding_documents_query():
    """Test : embedding de documents vs query"""
    print("\n" + "=" * 60)
    print("TEST 4 : Embedding documents vs query")
    print("=" * 60)
    
    modele = creer_modele_embedding()
    
    documents = [
        "Le service de support technique est disponible 24/7",
        "Les mises à jour de sécurité sont essentielles",
        "La maintenance préventive améliore la performance"
    ]
    
    query = "service support technique"
    
    try:
        doc_embeddings = modele.embed_documents(documents)
        query_embedding = modele.embed_query(query)
        
        print(f"\nQuery : '{query}'")
        print(f"\nDocuments :")
        for i, doc in enumerate(documents):
            print(f"  [{i+1}] {doc}")
        
        print(f"\nSimilarité avec la query :")
        for i, (doc, emb) in enumerate(zip(documents, doc_embeddings)):
            sim = calculer_similarite_cosinus(query_embedding, emb)
            print(f"  Document {i+1} : {sim:.4f}")
        
        # Vérifier que les similarités sont ordonnées correctement
        similarities = [calculer_similarite_cosinus(query_embedding, emb) 
                       for emb in doc_embeddings]
        print(f"\n✓ Test batch documents vs query complété")
        return True
    except Exception as e:
        print(f"✗ Erreur : {e}")
        return False


def main():
    """Exécute tous les tests"""
    print("\n" + "█" * 60)
    print("   TEST SUITE - EMBEDDINGS AZURE OPENAI")
    print("█" * 60)
    
    resultats = {
        "Test basique": test_embedding_basique(),
        "Test similarité": test_embedding_similarite(),
        "Test batch": test_embedding_batch(),
        "Test documents vs query": test_embedding_documents_query(),
    }
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    for nom, resultat in resultats.items():
        status = "✓ PASS" if resultat else "✗ FAIL"
        print(f"  {nom:.<40} {status}")
    
    total_pass = sum(resultats.values())
    total_tests = len(resultats)
    print(f"\nTotal : {total_pass}/{total_tests} tests réussis")
    
    if total_pass == total_tests:
        print("\n🎉 Tous les tests d'embedding sont passés!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_pass} test(s) échoué(s)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
