# Assistant RAG de Recommandation de Services

Un assistant intelligent qui trouve et recommande des services à partir de vos catalogues, en comprenant les questions en langage naturel.

## ⚡ Démarrage rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Configurer les clés API dans .env

# 3. Déposer les catalogues
# Ajouter vos fichiers pdf dans data/raw/

# 4. Lancer l'assistant
python app.py
```

## Configuration

Créez ou mettez à jour le fichier `.env` avec vos clés Azure OpenAI :

```env
OPENAI_API_KEY=votre_cle_chat
OPENAI_API_ENDPOINT=https://votre-ressource.openai.azure.com/
OPENAI_EMBEDDING_API_KEY=votre_cle_embedding
OPENAI_EMBEDDING_API_ENDPOINT=https://votre-ressource-embedding.openai.azure.com/
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

## 📁 Structure du projet

```
RAG_ASSISTANT/
├── src/                      # Cœur du système
│   ├── config.py            # Configuration centralisée
│   ├── ingestion.py         # Import des catalogues
│   ├── retrieval.py         # Recherche dans la base
│   ├── generation.py        # Génération des réponses
│   └── utils.py             # Fonctions partagées
│
├── data/
│   ├── raw/                 # Déposer les catalogues ici
│   └── processed/           # Catalogues indexés
│
├── vector_db/               # Base ChromaDB (auto-générée)
├── prompts/                 # Instruction de l'assistant
├── ui/                      # Interface Streamlit
├── tests/                   # Tests et utilitaires
├── app.py                   # Point d'entrée principal
└── config.yaml              # Configuration générale
```

## Utilisation

**Première utilisation :** La base vectorielle se crée automatiquement au premier lancement.

```bash
python app.py
```

**Après ajout de catalogues :** Réindexer la base :

```bash
python tests/reindexer.py
```

##  Tests

Lancer les cas de test :

```bash
python tests/test_scenarios.py
```

Voir [scenarios/test_cases.csv](scenarios/test_cases.csv) pour les tests disponibles.

## Comment ça marche

1. **Ingestion** → Les catalogues sont chargés et découpés en chunks
2. **Vectorisation** → Chaque chunk devient un vecteur (embedding)
3. **Stockage** → Les vecteurs sont indexés dans ChromaDB
4. **Recherche** → Une question pose cherche les documents similaires
5. **Génération** → GPT génère une recommandation basée sur les documents trouvés
6. **Mémoire** → L'assistant se souvient des 10 derniers échanges

## ⚙️ Comportement de l'assistant

- Recommande **1 à 2 services** par réponse
- Retient l'**historique** de la conversation (10 échanges)
- Refuse poliment les questions **hors catalogue**
- Reformule les questions **ambiguës** en tenant compte du contexte

## Dépannage

**Les recommandations ne sont pas bonnes ?**

- Vérifier que les catalogues sont dans `data/raw/`
- Relancer `python tests/reindexer.py`

**Erreur de clés API ?**

- Vérifier le fichier `.env`
- Les clés doivent correspondre à une ressource Azure OpenAI valide

**La base vectorielle ne se met pas à jour ?**

- Supprimer le dossier `vector_db/`
- Relancer l'application
