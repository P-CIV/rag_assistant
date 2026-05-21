import os
import sys
import shutil
from datetime import datetime

# ajouter le chemin du projet pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# charger les variables d'environnement
from dotenv import load_dotenv
from src.config import (
    DOSSIER_BASE_VECTEURS,
    DOSSIER_DONNEES_BRUTES,
    DOSSIER_DONNEES_TRAITEES,
)
from src.ingestion import (
    charger_documents,
    decouper_documents,
    creer_base_vecteurs,
    sauvegarder_documents_bruts,
    sauvegarder_chunks,
)

load_dotenv()


def reindexer():
    # afficher le titre
    print("=" * 80)
    print("REINDEXATION DE LA BASE VECTORIELLE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    try:
        # supprimer l'ancienne base vectorielle
        if os.path.exists(DOSSIER_BASE_VECTEURS):
            print(f"Suppression de la base vectorielle existante...")
            shutil.rmtree(DOSSIER_BASE_VECTEURS)
            print("Base supprimée.\n")

        # supprimer les données traitées précédentes
        if os.path.exists(DOSSIER_DONNEES_TRAITEES):
            print("Suppression des données traitées...")
            shutil.rmtree(DOSSIER_DONNEES_TRAITEES)
            print("Données supprimées.\n")

        # charger tous les PDFs
        print("Chargement des catalogues...")
        documents = charger_documents(DOSSIER_DONNEES_BRUTES)

        # sauvegarder le texte brut extrait
        print("\nSauvegarde du texte brut...")
        sauvegarder_documents_bruts(documents, DOSSIER_DONNEES_TRAITEES)

        # découper les documents en chunks
        print("\nDécoupe des documents...")
        morceaux = decouper_documents(documents)

        # sauvegarder les chunks individuels
        print("\nSauvegarde des chunks...")
        chemin_chunks = os.path.join(DOSSIER_DONNEES_TRAITEES, "chunks")
        os.makedirs(chemin_chunks, exist_ok=True)
        sauvegarder_chunks(morceaux, chemin_chunks)

        # créer et indexer la nouvelle base vectorielle
        print("\nCréation de la base vectorielle...")
        base = creer_base_vecteurs(morceaux, DOSSIER_BASE_VECTEURS)

        # afficher le résumé
        print("\n" + "=" * 80)
        print("REINDEXATION TERMINÉE AVEC SUCCÈS")
        print("=" * 80)
        total = base._collection.count()
        print(f"Total chunks indexés: {total}\n")

    # fichier non trouvé
    except FileNotFoundError as e:
        print(f"\nERREUR: {e}")
        sys.exit(1)
    # interruption utilisateur
    except KeyboardInterrupt:
        print("\n\nReindexation interrompue.")
        sys.exit(130)
    # autres erreurs
    except Exception as e:
        print(f"\nERREUR: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# point d'entrée
if __name__ == "__main__":
    reindexer()
