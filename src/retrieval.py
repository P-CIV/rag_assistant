import re
from langchain_chroma import Chroma
from src.config import NOMBRE_DOCS_RECUPERES

# M1 — Seuil de distance cosinus maximum (0 = identique, 2 = opposé)
# 0.7 correspond à ~65% de confiance minimum
SEUIL_DISTANCE_MAX = 0.7


def creer_retriever(base: Chroma):
    # Crée un retriever pour la recherche vectorielle
    return base.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": NOMBRE_DOCS_RECUPERES, "score_threshold": 0.3},
    )


def extraire_signal_recherche(reponse_brute: str) -> tuple:
    """Détecte et extrait le signal [RECHERCHE: ...] de la réponse du LLM"""
    match = re.search(r"\[RECHERCHE:\s*([^\]]+)\]", reponse_brute, re.IGNORECASE)
    if match:
        query = match.group(1).strip()
        texte_restant = reponse_brute[: match.start()] + reponse_brute[match.end() :]
        texte_restant = texte_restant.strip()
        return query, texte_restant
    return None, reponse_brute


def recuperer_documents(query: str, base: Chroma) -> list:
    """Effectue une recherche vectorielle et filtre les résultats peu pertinents"""
    resultats = base.similarity_search_with_score(query, k=NOMBRE_DOCS_RECUPERES)
    # M1 — Appliquer le seuil : ne garder que les documents suffisamment proches
    resultats_filtres = [
        (doc, score) for doc, score in resultats if score <= SEUIL_DISTANCE_MAX
    ]
    return resultats_filtres if resultats_filtres else resultats[:1]
