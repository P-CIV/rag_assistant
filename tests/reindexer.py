import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.config import DOSSIER_BASE_VECTEURS
from src.ingestion import charger_documents, decouper_documents, creer_base_vecteurs

load_dotenv()


def reindexer():
    # Supprime l'ancienne base et recrée une nouvelle depuis les catalogues
    if os.path.exists(DOSSIER_BASE_VECTEURS):
        shutil.rmtree(DOSSIER_BASE_VECTEURS)
        print("Ancienne base vectorielle supprimee.")

    from src.config import DOSSIER_DONNEES_BRUTES
    documents = charger_documents(DOSSIER_DONNEES_BRUTES)
    morceaux = decouper_documents(documents)
    creer_base_vecteurs(morceaux, DOSSIER_BASE_VECTEURS)
    print("Reindexation terminee avec succes.")


if __name__ == "__main__":
    reindexer()
