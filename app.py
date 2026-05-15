from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import re

from src.ingestion import executer_ingestion, charger_base_existante
from src.retrieval import extraire_signal_recherche, recuperer_documents
from src.generation import premiere_passe, deuxieme_passe, repondre
from src.config import MAX_HISTORIQUE, DOSSIER_BASE_VECTEURS
import os

load_dotenv()


def lancer_assistant():
    """Lance la boucle de conversation avec l'assistant"""
    print("Chargement de l'assistant de recommandation...")

    try:
        if os.path.exists(DOSSIER_BASE_VECTEURS) and os.listdir(DOSSIER_BASE_VECTEURS):
            base = charger_base_existante(DOSSIER_BASE_VECTEURS)
        else:
            base = executer_ingestion()
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return

    historique = []
    print("\nBonjour ! Je suis votre assistant de recommandation.")
    print("Je suis là pour vous aider à trouver le service qui vous convient.")
    print("Tapez 'quitter' pour terminer la conversation.\n")

    while True:
        question = input("Vous : ").strip()

        if not question:
            continue

        if question.lower() in ("quitter", "exit", "quit"):
            print("\nMerci pour votre visite. A bientôt !")
            break

        # Utiliser la fonction repondre() qui retourne réponse + sources
        resultat = repondre(question, historique)
        reponse_finale = resultat["reponse"]
        sources = resultat["sources"]

        print(f"\nAssistant : {reponse_finale}")

        # Afficher les sources si disponibles
        if sources:
            print("\n📚 Sources utilisées :")
            for i, source in enumerate(sources, 1):
                fichier = os.path.basename(source.get("fichier", "Inconnu"))
                score = source.get("score", 0)
                print(f"  [{i}] {fichier} (confiance: {score}%)")

        print()

        historique.append(HumanMessage(content=question))
        historique.append(AIMessage(content=reponse_finale))

        if len(historique) > MAX_HISTORIQUE * 2:
            historique = historique[-(MAX_HISTORIQUE * 2) :]


if __name__ == "__main__":
    lancer_assistant()
