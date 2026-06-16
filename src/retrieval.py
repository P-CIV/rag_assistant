import re
import os
import logging
from src.qdrant_wrapper import QdrantRetrieverWrapper
from src.config import NOMBRE_DOCS_RECUPERES, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

# Seuil de distance cosinus maximum 
SEUIL_DISTANCE_MAX = 0.7


def creer_retriever(base: QdrantRetrieverWrapper):
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


def recuperer_documents(query: str, base: QdrantRetrieverWrapper) -> list:
    """Effectue une recherche vectorielle Qdrant et filtre les résultats peu pertinents"""
    resultats = base.similarity_search_with_score(query, k=NOMBRE_DOCS_RECUPERES)
    resultats_filtres = [
        (doc, score) for doc, score in resultats if score <= SEUIL_DISTANCE_MAX
    ]
    return resultats_filtres if resultats_filtres else resultats[:1]


def charger_base_qdrant() -> QdrantRetrieverWrapper:
    """Charge et retourne le wrapper Qdrant connecté à la collection cloud."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise EnvironmentError(
            "QDRANT_URL et QDRANT_API_KEY sont requis. Vérifie ton .env"
        )
    logger.info(f"Connexion à Qdrant Cloud — collection '{QDRANT_COLLECTION}'")
    return QdrantRetrieverWrapper(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION,
    )
