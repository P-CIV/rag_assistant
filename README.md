# Assistant RAG de Recommandation de Services

Un assistant qui cherche et recommande des services dans vos catalogues en comprenant vos questions.

## Démarrage rapide

```bash
# Créer un environement virtuel
python -m venv venv

# Activer l'environnement

# Installer les dépendances
pip install -r requirements.txt

# Configurer les clés API
# Créer/éditer le fichier .env 

# Ajouter vos catalogues
# Placer les fichiers PDF dans data/raw/

# Lancer l'assistant
python app.py
```

## Configuration

Créez un fichier `.env` avec vos clés API :

```env
OPENAI_API_KEY=votre_cle
OPENAI_API_ENDPOINT=https://votre-ressource.openai.azure.com/
OPENAI_EMBEDDING_API_KEY=votre_cle_embedding
OPENAI_EMBEDDING_API_ENDPOINT=https://votre-ressource-embedding.openai.azure.com/
AZURE_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-...
```

## Structure du projet

```
RAG_ASSISTANT/
├── src/                      # Code principal
│   ├── config.py            # Configuration Azure OpenAI
│   ├── ingestion.py         # Charge PDFs, découpe chunks, vectorise
│   ├── retrieval.py         # Recherche vectorielle (Chroma)
│   ├── generation.py        # Génération LLM (2 passes)
│   └── utils.py             # Utilitaires (embeddings)
│
├── data/
│   ├── raw/                 # Vos catalogues (PDF)
│   └── processed/           # Textes extraits + chunks
│
├── scenarios/               # Cas de test
│   └── test_cases.csv       # scénarios des questions de recommandation 
│
├── vector_db/               # ChromaDB (auto-créée)
├── config.yaml              # Paramètres (chunk size, temperature, etc...)
├── prompts/                 # Instructions pour le LLM
│   └── rag_prompt.txt       # Prompt système
├── app.py                   # CLI interactive
├── server_fastapi.py        # API REST
├── requirements.txt         # Dépendances
└── README.md               # Ce fichier
```

## Utilisation

**Premier lancement :**

```bash
python app.py
```

La base vectorielle se crée automatiquement.

**Après ajout de catalogues :**

```bash
python reindexer.py
```

Recrée la base depuis les nouveaux PDFs.

**Lancer les tests :**

```bash
python evaluator.py
```

Évalue les scénarios et génère un rapport JSON.

## Comment ça marche

L'assistant suit ces étapes pour répondre à vos questions :

1. **Ingestion** Les catalogues PDF sont chargés et découpés
2. **Vectorisation** Chaque morceau devient un vecteur (embedding)
3. **Indexation** Les vecteurs sont stockés dans ChromaDB
4. **Recherche** Votre question trouve les documents pertinents
5. **Génération structurée** Appel Azure OpenAI GPT-4.1-mini (temperature 0.3) en deux passes : détection d'intention puis génération enrichie avec contexte et parsing structuré
6. **Mémoire** L'assistant se souvient des 10 derniers échanges

## Caractéristiques

- Recommande 1 à 2 services par réponse
- Mémorise l'historique (10 échanges)
- Refuse les questions hors catalogue
- Reformule les questions ambiguës avec le contexte
- Génération à 2 passes avec parsing de réponse

## Tests

scénarios couvrant :

- **Recommandations** : Questions sur les services (matériel, serveurs, RGPD, maintenance)
- **Refus polies** : Questions hors catalogue
- **Suivi historique** : Questions avec contexte

Résultats stockés dans : `evaluator.json`

## Dépannage

**Les recommandations ne sont pas bonnes ?**

- Vérifier que les catalogues sont dans `data/raw/`
- Relancer `python reindexer.py`

**Erreur de clés API ?**

- Vérifier le fichier `.env`
- Les clés doivent être valides pour Azure OpenAI

**La base n'est pas à jour ?**

- Supprimer le dossier `vector_db/`
- Relancer l'application
