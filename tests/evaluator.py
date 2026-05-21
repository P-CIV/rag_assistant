import os
import sys
import csv
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.generation import repondre

load_dotenv()


class EvaluatorRAG:
    def __init__(self, fichier_scenarios: str = "scenarios/test_cases.csv"):
        self.fichier_scenarios = fichier_scenarios
        self.cas_de_test = []
        self.resultats = []

    def charger_scenarios(self):
        # charge les scénarios du CSV
        with open(self.fichier_scenarios, encoding="utf-8") as f:
            self.cas_de_test = list(csv.DictReader(f))
        print(f"Chargé {len(self.cas_de_test)} scénarios\n")

    def evaluer(self, reponse_data, cas_attendu):
        # évalue si la réponse est correcte
        type_attendu = cas_attendu.get("reponse_attendue_type", "").strip()
        nb_sources = reponse_data["nb_sources"]
        reponse = reponse_data["reponse"]

        if type_attendu == "recommandation":
            correct = nb_sources > 0
        elif type_attendu == "refus_poli":
            correct = nb_sources == 0
        elif type_attendu == "suivi_historique":
            correct = any(
                mot in reponse.lower() for mot in ["dernier", "suggéré", "proposé"]
            )
        else:
            correct = nb_sources > 0

        return {"correct": correct, "type": type_attendu, "nb_sources": nb_sources}

    def executer_tests(self):
        # exécute tous les tests
        print("=" * 80)
        print("ÉVALUATION RAG")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().isoformat()}\n")

        historique = []

        for i, cas in enumerate(self.cas_de_test, 1):
            question = cas["question"]
            print(f"Test {i}/{len(self.cas_de_test)}: {question[:55]}...", end=" ")

            try:
                # exécuter le test
                temps_debut = time.time()
                reponse_data = repondre(question, historique=historique)
                temps_exec = time.time() - temps_debut

                # évaluer
                eval_result = self.evaluer(reponse_data, cas)

                # enregistrer
                resultat = {
                    "id": i,
                    "question": question,
                    "type": eval_result["type"],
                    "correct": eval_result["correct"],
                    "nb_sources": eval_result["nb_sources"],
                    "temps_s": round(temps_exec, 3),
                }
                self.resultats.append(resultat)

                # afficher
                status = "OK" if eval_result["correct"] else "KO"
                print(f"[{status}] {eval_result['type']}")

                # ajouter à l'historique
                historique.append(HumanMessage(content=question))
                historique.append(AIMessage(content=reponse_data.get("reponse", "")))

            except Exception as e:
                print(f"[ERR] {str(e)}")
                self.resultats.append(
                    {
                        "id": i,
                        "question": question,
                        "correct": False,
                        "erreur": str(e),
                    }
                )

    def afficher_metriques(self):
        # affiche les métriques globales
        if not self.resultats:
            return

        total = len(self.resultats)
        corrects = sum(1 for r in self.resultats if r.get("correct", False))
        temps_total = sum(r.get("temps_s", 0) for r in self.resultats)
        accuracy = round(corrects / total * 100, 1) if total > 0 else 0

        # par type
        par_type = {}
        for r in self.resultats:
            type_r = r.get("type", "unknown")
            if type_r not in par_type:
                par_type[type_r] = {"total": 0, "correct": 0}
            par_type[type_r]["total"] += 1
            if r.get("correct", False):
                par_type[type_r]["correct"] += 1

        print("\n" + "=" * 80)
        print("RÉSULTATS")
        print("=" * 80)
        print(f"Accuracy: {corrects}/{total} ({accuracy}%)")
        print(f"Temps total: {round(temps_total, 2)}s")
        print(f"Temps moyen: {round(temps_total / total, 3)}s")

        print("\nPar type:")
        for type_r, stats in par_type.items():
            pct = (
                round(stats["correct"] / stats["total"] * 100, 1)
                if stats["total"] > 0
                else 0
            )
            print(f"  {type_r}: {stats['correct']}/{stats['total']} ({pct}%)")

        self.exporter_json(accuracy, par_type)

    def exporter_json(self, accuracy, par_type):
        # exporte les résultats en JSON
        os.makedirs("tests/rapports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fichier = os.path.join("tests/rapports", f"evaluator_{timestamp}.json")

        rapport = {
            "timestamp": datetime.now().isoformat(),
            "accuracy_pct": accuracy,
            "performance_par_type": par_type,
            "resultats": self.resultats,
        }

        with open(fichier, "w", encoding="utf-8") as f:
            json.dump(rapport, f, indent=2, ensure_ascii=False)

        print(f"\nRapport: {fichier}\n")

    def executer(self):
        # pipeline complète
        try:
            self.charger_scenarios()
            self.executer_tests()
            self.afficher_metriques()
        except KeyboardInterrupt:
            print("\n\nInterrompu")
            sys.exit(130)
        except Exception as e:
            print(f"\nERREUR: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    EvaluatorRAG().executer()
