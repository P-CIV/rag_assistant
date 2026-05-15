import os
import re
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import FICHIER_PROMPT, MAX_RECOMMANDATIONS, DOSSIER_BASE_VECTEURS
from src.utils import creer_modele_chat, lire_prompt, formater_documents
from src.retrieval import extraire_signal_recherche, recuperer_documents
from src.ingestion import executer_ingestion, charger_base_existante

# F1 — Cache du prompt système (lecture disque unique au lieu de 2 I/O par échange)
_prompt_cache = None


def _get_prompt() -> str:
    """Retourne le prompt système mis en cache (lecture disque unique)."""
    global _prompt_cache
    if _prompt_cache is None:
        try:
            _prompt_cache = lire_prompt(FICHIER_PROMPT)
        except FileNotFoundError:
            _prompt_cache = _prompt_par_defaut()
    return _prompt_cache


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
        f"Question du client : {question}\n\n" f"Extraits du catalogue :\n{contexte}"
    )

    messages = construire_messages(prompt_systeme, historique, contenu_utilisateur)
    resultat = modele.invoke(messages)
    return resultat.content


def _prompt_par_defaut() -> str:
    return (
        f"Tu es un assistant de recommandation chaleureux et professionnel. "
        f"Tu aides les clients à trouver les services qui correspondent le mieux à leurs besoins "
        f"en te basant uniquement sur les catalogues fournis. "
        f"Tu proposes au maximum {MAX_RECOMMANDATIONS} service(s) par réponse. "
        f"Si une recherche dans le catalogue est nécessaire, commence ta réponse par "
        f"[RECHERCHE: <question reformulée>] puis continue. "
        f"Sinon, réponds directement. "
        f"Tu t'exprimes en français, de façon naturelle et chaleureuse."
    )


def repondre(question: str, historique: list = None) -> dict:
    """Orchestre la pipeline RAG: charge la base, décide si recherche nécessaire, retourne réponse + sources"""
    if historique is None:
        historique = []

    try:
        if os.path.exists(DOSSIER_BASE_VECTEURS) and os.listdir(DOSSIER_BASE_VECTEURS):
            base = charger_base_existante(DOSSIER_BASE_VECTEURS)
        else:
            print(" Création de la base vectorielle (première utilisation)...")
            base = executer_ingestion()
    except Exception as e:
        return {
            "reponse": f" Erreur lors du chargement de la base vectorielle: {e}",
            "nb_sources": 0,
            "sources": [],
        }

    reponse_brute = premiere_passe(question, historique)
    query_recherche, reponse_finale = extraire_signal_recherche(reponse_brute)
    sources = []

    if query_recherche:
        try:
            # Récupérer les documents pertinents avec scores
            resultats = recuperer_documents(query_recherche, base)

            # Séparer les documents et les scores
            documents = [doc for doc, score in resultats]

            # Deuxième passe avec contexte (documents seulement, pas les scores)
            reponse_finale = deuxieme_passe(question, documents, historique)

            # Extraire les métadonnées des sources avec les scores
            sources = []
            for doc, score in resultats:
                try:
                    # ChromaDB avec hnsw:space=cosine retourne une distance cosinus
                    # (0 = identique, 2 = opposé). Formule de conversion correcte :
                    # confiance = (1 - distance/2) * 100
                    if isinstance(score, (int, float)):
                        score_confiance = round((1 - score / 2) * 100, 1)
                    else:
                        score_confiance = 0
                except Exception as e:
                    score_confiance = 0

                score_confiance = max(0, min(100, score_confiance))

                source_info = {
                    "fichier": doc.metadata.get("source", "Inconnu"),
                    "service": doc.metadata.get("service", "N/A"),
                    "score": score_confiance,
                }
                sources.append(source_info)

            # M2 — Seuil de confiance minimum : si la moyenne < 70%, ignorer les sources
            if sources:
                confiance_moyenne = sum(s["score"] for s in sources) / len(sources)
                if confiance_moyenne < 70:
                    sources = []

        except Exception as e:
            reponse_finale = f"{reponse_finale}\n\n⚠️ Erreur lors de la recherche: {e}"

    reponse_finale = re.sub(
        r"\[RECHERCHE:[^\]]*\]", "", reponse_finale, flags=re.IGNORECASE
    ).strip()

    return {"reponse": reponse_finale, "nb_sources": len(sources), "sources": sources}
