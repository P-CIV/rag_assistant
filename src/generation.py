import os
import re
import logging
import dotenv
dotenv.load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from src.config import FICHIER_PROMPT, MAX_RECOMMANDATIONS
from src.utils import creer_modele_chat, lire_prompt, formater_documents
from src.retrieval import extraire_signal_recherche, recuperer_documents, charger_base_qdrant

logger = logging.getLogger(__name__)

# Cache singleton du prompt et de la base Qdrant
_prompt_cache = None
_base_cache = None


def _get_prompt() -> str:
    """Retourne le prompt système mis en cache (lecture disque unique)."""
    global _prompt_cache
    if _prompt_cache is None:
        try:
            _prompt_cache = lire_prompt(FICHIER_PROMPT)
        except FileNotFoundError:
            _prompt_cache = _prompt_par_defaut()
    return _prompt_cache


def _get_base():
    """Retourne la connexion Qdrant mise en cache (connexion unique au démarrage)."""
    global _base_cache
    if _base_cache is None:
        _base_cache = charger_base_qdrant()
    return _base_cache


def construire_messages(
    prompt_systeme: str, historique: list, contenu_utilisateur: str
) -> list:
    """Assemble la liste complète des messages pour le modèle"""
    messages = [SystemMessage(content=prompt_systeme)]
    messages.extend(historique)
    messages.append(HumanMessage(content=contenu_utilisateur))
    return messages


def premiere_passe(question: str, historique: list) -> str:
    """Première passe: le LLM décide s'il doit chercher dans les catalogues"""
    modele = creer_modele_chat()
    prompt_systeme = _get_prompt()
    messages = construire_messages(prompt_systeme, historique, question)
    resultat = modele.invoke(messages)
    return resultat.content


def deuxieme_passe(question: str, documents: list, historique: list) -> str:
    """Deuxième passe: répond en enrichissant avec les documents récupérés"""
    modele = creer_modele_chat()
    prompt_systeme = _get_prompt()

    contexte = formater_documents(documents)
    contenu_utilisateur = (
        f"Question du client : {question}\n\n"
        f"Extraits du catalogue :\n{contexte}"
    )

    messages = construire_messages(prompt_systeme, historique, contenu_utilisateur)
    resultat = modele.invoke(messages)
    return resultat.content


def _prompt_par_defaut() -> str:
    return (
        """Tu es un assistant virtuel professionnel spécialisé dans les services informatiques.

          Ton rôle est :
        - comprendre le besoin utilisateur,
        - répondre de manière naturelle,
        - recommander uniquement des services réellement présents dans les documents fournis.

        RÈGLES ABSOLUES :

        1. Utilise UNIQUEMENT les informations présentes dans le contexte fourni ci-dessous.
        2. N'invente jamais un service.
        3. N'invente jamais une source.
        4. Si aucun document pertinent n'est fourni, réponds poliment que tu ne disposes pas d'information suffisante.
        5. Si le message est une simple salutation, réponds naturellement sans parler de services.
        6. Ne mentionne jamais de source si aucun document pertinent n'est disponible.
        7. Sois fluide, humain, professionnel et concis.
        8. À la fin de ta réponse, ajoute obligatoirement une section 'Sources utilisées' sous cette forme exacte :
          - [Nom de la source] (Page du document) - Catégorie : [Nom de la catégorie]
        9. Si la question de l'utilisateur est une salutation ou tout autre choses n'étant pas dans le contexte, reponds juste a la salutation sans mettre de source quelconque et de services
        10. Si la question de l'utilisateur est dans le contexte repond en 6 à 7 phrases maximum"""
    )


def repondre(question: str, historique: list = None) -> dict:
    """Orchestre la pipeline RAG: charge la base Qdrant, décide si recherche nécessaire, retourne réponse + sources"""
    if historique is None:
        historique = []

    try:
        base = _get_base()
    except Exception as e:
        logger.error(f"Erreur chargement base Qdrant : {e}")
        return {
            "reponse": f"Erreur lors du chargement de la base vectorielle: {e}",
            "nb_sources": 0,
            "sources": [],
        }

    reponse_brute = premiere_passe(question, historique)
    query_recherche, reponse_finale = extraire_signal_recherche(reponse_brute)
    sources = []

    if query_recherche:
        try:
            resultats = recuperer_documents(query_recherche, base)
            documents = [doc for doc, score in resultats]
            reponse_finale = deuxieme_passe(question, documents, historique)

            sources = []
            for doc, score in resultats:
                try:
                    # score ici est une distance cosinus (0=identique, 1=opposé)
                    if isinstance(score, (int, float)):
                        score_confiance = round((1 - score) * 100, 1)
                    else:
                        score_confiance = 0
                except Exception:
                    score_confiance = 0

                score_confiance = max(0, min(100, score_confiance))

                source_info = {
                    "fichier": doc.metadata.get("source", "Inconnu"),
                    "service": doc.metadata.get("service", "N/A"),
                    "score": score_confiance,
                }
                sources.append(source_info)

            # Seuil de confiance minimum : si la moyenne < 70%, ignorer les sources
            if sources:
                confiance_moyenne = sum(s["score"] for s in sources) / len(sources)
                if confiance_moyenne < 70:
                    sources = []

        except Exception as e:
            reponse_finale = f"{reponse_finale}\n\nErreur lors de la recherche: {e}"

    reponse_finale = re.sub(
        r"\[RECHERCHE:[^\]]*\]", "", reponse_finale, flags=re.IGNORECASE
    ).strip()

    return {"reponse": reponse_finale, "nb_sources": len(sources), "sources": sources}
