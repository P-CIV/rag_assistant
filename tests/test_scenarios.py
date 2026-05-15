import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.generation import repondre

load_dotenv()


def executer_tests(fichier_scenarios: str = "scenarios/test_cases.csv"):
    """Charge la base et joue chaque cas de test du fichier CSV.

    B1 — Correction : utilise repondre() qui orchestre la pipeline complète
    (premiere_passe + recuperer_documents + deuxieme_passe) avec la bonne
    signature, plutôt que generer_reponse() (inexistante) et
    recuperer_documents(question, historique, base) (mauvaise signature).
    """
    print("Initialisation de la base vectorielle pour les tests...")

    with open(fichier_scenarios, newline="", encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        cas_de_test = list(lecteur)

    print(f"\n{len(cas_de_test)} cas de test trouves.\n")
    historique = []

    for i, cas in enumerate(cas_de_test, 1):
        question = cas["question"]
        hors_catalogue = cas.get("hors_catalogue", "non").strip().lower() == "oui"
        print(f"Test {i} : {question}")
        print(f"  Type attendu : {cas.get('reponse_attendue_type', '?')}"
              f"{'  [hors catalogue]' if hors_catalogue else ''}")

        resultat = repondre(question, historique=historique)
        reponse = resultat["reponse"]

        historique.append(HumanMessage(content=question))
        historique.append(AIMessage(content=reponse))

        print(f"  Sources utilisées : {resultat['nb_sources']}")
        print(f"Reponse : {reponse}")
        print("-" * 60)


if __name__ == "__main__":
    executer_tests()
