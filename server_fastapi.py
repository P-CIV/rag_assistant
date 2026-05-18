import sys
import os
import re
import uuid
import random
import logging
import secrets
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

# Imports FastAPI
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Charger les variables d'environnement
from dotenv import load_dotenv

load_dotenv()

# Ajouter le dossier src au chemin Python pour les imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
if os.path.exists(SRC_PATH) and SRC_PATH not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Importer le module RAG avec gestion d'erreur
try:
    from src.generation import repondre
    from langchain_core.messages import HumanMessage, AIMessage

    RAG_DISPONIBLE = True
except ImportError as e:
    logging.warning(f"Module RAG non disponible : {e}")
    RAG_DISPONIBLE = False

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("kova_api")

# Constantes de sécurité et limites
MAX_HISTORIQUE = 10
TTL_SESSION_MINUTES = 60
MAX_SESSIONS_TOTALES = 500
MAX_MESSAGE_CHARS = 2000
RATE_LIMIT_CHAT = "10/minute"
RATE_LIMIT_SESSION = "20/minute"

# Timeout RAG — augmenté pour couvrir le cold start Render Free (~30-60 s)
RAG_TIMEOUT_SECONDES = 180.0

# ThreadPoolExecutor limité à 2 workers (contrainte RAM Render Free 512 MB)
_executor = ThreadPoolExecutor(max_workers=2)

# Configurer les origines CORS autorisées
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS: list[str] = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["http://localhost:5500", "http://127.0.0.1:5500"]
)

# Authentification par clé API (optionnelle)
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verifier_api_key(key: Optional[str] = Depends(api_key_header)) -> bool:
    """Vérifie la clé API si définie, sinon laisse passer."""
    if not API_SECRET_KEY:
        return True
    if not key:
        raise HTTPException(status_code=401, detail="Clé API manquante.")
    # Comparaison à temps constant pour éviter les timing attacks
    if not secrets.compare_digest(key, API_SECRET_KEY):
        raise HTTPException(status_code=403, detail="Clé API invalide.")
    return True


# Initialiser le rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])

# Créer l'application FastAPI
app = FastAPI(
    title="KOVA IA API",
    description="Backend RAG pour l'assistant de recommandation KOVA",
    version="2.0.0",
    docs_url="/docs" if os.getenv("ENV", "production") == "development" else None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Stockage des sessions en mémoire
sessions: dict[str, dict] = {}

_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def nettoyer_sessions_expirées() -> None:
    """Supprime les sessions inactives depuis plus de TTL_SESSION_MINUTES."""
    expiration = datetime.utcnow() - timedelta(minutes=TTL_SESSION_MINUTES)
    ids_expirés = [
        sid for sid, data in sessions.items() if data["dernière_activité"] < expiration
    ]
    for sid in ids_expirés:
        del sessions[sid]
    if ids_expirés:
        logger.info(f"{len(ids_expirés)} session(s) expirée(s) supprimée(s)")


# Middleware de nettoyage périodique (1 requête sur 50)
@app.middleware("http")
async def nettoyage_periodique(request: Request, call_next):
    """Déclenche le nettoyage des sessions expirées de façon aléatoire."""
    if random.randint(1, 50) == 1:
        nettoyer_sessions_expirées()
    return await call_next(request)


# Wrapper async pour appels RAG bloquants avec timeout
async def appeler_rag(question: str, historique: list) -> dict:
    """Exécute repondre() dans un thread et impose un timeout de RAG_TIMEOUT_SECONDES."""
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(
            _executor,
            lambda: repondre(question=question, historique=historique),
        ),
        timeout=RAG_TIMEOUT_SECONDES,
    )


# Warm-up au démarrage pour éviter le cold start sur la première requête
@app.on_event("startup")
async def warmup():
    """Pré-charge le pipeline RAG au démarrage."""
    if not RAG_DISPONIBLE:
        return
    logger.info("Warm-up RAG en cours…")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            lambda: repondre(question="bonjour", historique=[]),
        )
        logger.info("Warm-up RAG terminé.")
    except Exception as e:
        logger.warning(f"Warm-up RAG échoué (non bloquant) : {e}")


# Modèles Pydantic


class SessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str

    @field_validator("session_id")
    @classmethod
    def valider_uuid(cls, v: str) -> str:
        """Valide que session_id est un UUID v4."""
        if not _UUID_PATTERN.match(v):
            raise ValueError("session_id invalide.")
        return v

    @field_validator("message")
    @classmethod
    def valider_message(cls, v: str) -> str:
        """Valide et limite la taille du message."""
        v = v.strip()
        if not v:
            raise ValueError("Le message ne peut pas être vide.")
        if len(v) > MAX_MESSAGE_CHARS:
            raise ValueError(
                f"Message trop long ({len(v)} caractères, max {MAX_MESSAGE_CHARS})."
            )
        return v


class SourceInfo(BaseModel):
    fichier: str
    # Score normalisé en pourcentage
    score: float
    service: Optional[str] = None


class ChatResponse(BaseModel):
    reponse: str
    nb_sources: int
    sources: list[SourceInfo]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    rag_disponible: bool
    sessions_actives: int
    version: str


# Endpoints API


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Système"],
    dependencies=[Depends(verifier_api_key)],
)
async def health():
    """Vérifie que l'API et le pipeline RAG sont opérationnels."""
    nettoyer_sessions_expirées()
    return HealthResponse(
        status="ok",
        rag_disponible=RAG_DISPONIBLE,
        sessions_actives=len(sessions),
        version=app.version,
    )


@app.post(
    "/session/nouvelle",
    response_model=SessionResponse,
    tags=["Sessions"],
    dependencies=[Depends(verifier_api_key)],
)
@limiter.limit(RATE_LIMIT_SESSION)
async def créer_session(request: Request):
    """Crée une nouvelle session de conversation."""
    nettoyer_sessions_expirées()

    if len(sessions) >= MAX_SESSIONS_TOTALES:
        logger.warning("Limite de sessions atteinte — refus de création.")
        raise HTTPException(
            status_code=503,
            detail="Trop de sessions actives. Réessayez dans quelques minutes.",
        )

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "historique": [],
        "dernière_activité": datetime.utcnow(),
        "nb_échanges": 0,
    }
    logger.info(f"Nouvelle session créée : {session_id[:8]}…")
    return SessionResponse(
        session_id=session_id,
        message="Session créée avec succès",
    )


@app.delete(
    "/session/{session_id}",
    tags=["Sessions"],
    dependencies=[Depends(verifier_api_key)],
)
async def supprimer_session(session_id: str):
    """Supprime une session et libère sa mémoire."""
    if not _UUID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="session_id invalide.")

    if session_id in sessions:
        del sessions[session_id]
        logger.info(f"Session supprimée : {session_id[:8]}…")
        return {"message": "Session supprimée"}
    return {"message": "Session introuvable (déjà expirée ?)"}


# Endpoint POST pour fermeture propre via sendBeacon
@app.post(
    "/session/{session_id}/fermer",
    tags=["Sessions"],
    dependencies=[Depends(verifier_api_key)],
)
async def fermer_session(session_id: str):
    """Ferme proprement une session (compatible sendBeacon)."""
    if not _UUID_PATTERN.match(session_id):
        raise HTTPException(status_code=400, detail="session_id invalide.")
    sessions.pop(session_id, None)
    logger.info(f"Session fermée via sendBeacon : {session_id[:8]}…")
    return {"message": "Session fermée"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    dependencies=[Depends(verifier_api_key)],
)
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(request: Request, data: ChatRequest):
    """Envoie un message et retourne une réponse RAG avec sources."""
    if not RAG_DISPONIBLE:
        raise HTTPException(
            status_code=503,
            detail="Le pipeline RAG n'est pas disponible.",
        )

    if data.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session introuvable ou expirée. "
            "Créez une nouvelle session via POST /session/nouvelle.",
        )

    session = sessions[data.session_id]
    session["dernière_activité"] = datetime.utcnow()

    # Logger sans contenu utilisateur pour la RGPD
    logger.info(f"[{data.session_id[:8]}] Question reçue ({len(data.message)} chars)")

    # Appel RAG asynchrone avec timeout
    try:
        résultat = await appeler_rag(data.message, session["historique"])
    except asyncio.TimeoutError:
        logger.error(
            f"Timeout RAG [{data.session_id[:8]}] après {RAG_TIMEOUT_SECONDES} s"
        )
        raise HTTPException(
            status_code=503,
            detail="Le serveur démarre, veuillez réessayer dans 30 secondes.",
        )
    except Exception as e:
        # Log détaillé côté serveur, message générique côté client
        logger.error(f"Erreur RAG [{data.session_id[:8]}] : {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Erreur interne. Veuillez réessayer.",
        )

    réponse: str = résultat.get("reponse", "")
    sources_brutes: list = résultat.get("sources", [])
    nb_sources: int = résultat.get("nb_sources", 0)

    # Mettre à jour l'historique avec le nouvel échange
    session["historique"].append(HumanMessage(content=data.message))
    session["historique"].append(AIMessage(content=réponse))
    session["nb_échanges"] += 1

    # Troncature par paires pour ne jamais couper un échange Human/AI
    if len(session["historique"]) > MAX_HISTORIQUE * 2:
        limite = MAX_HISTORIQUE * 2
        brut = session["historique"][-limite:]
        # S'assurer qu'on commence toujours par un HumanMessage
        if brut and not isinstance(brut[0], HumanMessage):
            brut = brut[1:]
        session["historique"] = brut

    # Normalisation du score RAG en pourcentage lisible par le frontend
    sources = [
        SourceInfo(
            fichier=os.path.basename(src.get("fichier", "Inconnu")),
            score=round(src.get("score", 0) * 100, 1)
            if src.get("score", 0) <= 1.0
            else round(src.get("score", 0), 1),
            service=src.get("service"),
        )
        for src in sources_brutes
    ]

    logger.info(
        f"[{data.session_id[:8]}] Réponse générée — "
        f"{len(réponse)} chars, {nb_sources} source(s)"
    )

    return ChatResponse(
        reponse=réponse,
        nb_sources=nb_sources,
        sources=sources,
        session_id=data.session_id,
    )


# Point d'entrée
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server_fastapi:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
    )
