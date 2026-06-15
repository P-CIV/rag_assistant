import os
from dotenv import load_dotenv
load_dotenv()

# Configuration Azure OpenAI Chat
AZURE_CHAT_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_CHAT_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4.1-mini")
AZURE_CHAT_API_VERSION = os.getenv("AZURE_CHAT_API_VERSION", "2024-10-21")

# Configuration Azure OpenAI Embedding
AZURE_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY")
AZURE_EMBEDDING_ENDPOINT = os.getenv("OPENAI_EMBEDDING_API_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION", "2023-05-15")

# Chemins du projet
DOSSIER_DONNEES_BRUTES = "data/raw"
DOSSIER_DONNEES_TRAITEES = "data/processed"
DOSSIER_BASE_VECTEURS = "vector_db"
FICHIER_PROMPT = "prompts/rag_prompt.txt"

# Paramètres de découpage des documents
TAILLE_CHUNK = 1000
CHEVAUCHEMENT_CHUNK = 100

# Paramètres de récupération
NOMBRE_DOCS_RECUPERES = 4

# Nombre maximum de recommandations à proposer
MAX_RECOMMANDATIONS = 2

# Taille maximale de l'historique de conversation (en nombre d'échanges)
MAX_HISTORIQUE = 10
