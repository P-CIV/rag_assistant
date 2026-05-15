import os
import json
import time
from datetime import datetime
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import (
    DOSSIER_DONNEES_BRUTES,
    DOSSIER_DONNEES_TRAITEES,
    DOSSIER_BASE_VECTEURS,
    TAILLE_CHUNK,
    CHEVAUCHEMENT_CHUNK,
)
from src.utils import creer_modele_embedding


def nettoyer_texte(texte: str) -> str:
    """Nettoie et normalise le texte extrait des PDFs"""
    import re

    # Enlever les caractères corrompus
    texte = re.sub(r"[\x80-\xff]{3,}", "", texte)
    texte = re.sub(r"ǩ|¬|ǫ|ǅ|ǌ|\?{2,}", "", texte)

    # Normaliser les espaces
    texte = re.sub(r" {2,}", " ", texte)
    texte = re.sub(r"\n\n\n+", "\n\n", texte)

    # Nettoyer les lignes invalides
    lines = texte.split("\n")
    lines = [
        line.strip()
        for line in lines
        if line.strip() and not re.match(r"^[\.\-]+$", line.strip())
    ]

    # Enlever les nombres isolés sauf en contexte
    cleaned_lines = []
    for i, line in enumerate(lines):
        if re.match(r"^\d{1,4}$", line):
            prev_context = lines[i - 1] if i > 0 else ""
            next_context = lines[i + 1] if i < len(lines) - 1 else ""
            if not (
                any(
                    word in prev_context.lower()
                    for word in ["contrat", "heures", "ans", "kit"]
                )
            ):
                continue
        cleaned_lines.append(line)

    texte = "\n".join(cleaned_lines)
    texte = texte.strip()

    return texte


def charger_documents(dossier: str) -> list:
    # Charge tous les fichiers PDF depuis le dossier des catalogues
    if not os.path.exists(dossier):
        raise FileNotFoundError(
            f"Le dossier {dossier} n'existe pas. Ajoutez vos catalogues."
        )

    documents = []
    fichiers_pdf = [f for f in os.listdir(dossier) if f.endswith(".pdf")]

    if not fichiers_pdf:
        raise FileNotFoundError(f"Aucun fichier (.pdf) trouvé dans {dossier}.")

    print(f" {len(fichiers_pdf)} fichier(s) PDF trouvé(s)")

    # Chargement des fichiers PDF avec gestion d'erreur
    for i, fichier in enumerate(fichiers_pdf, 1):
        chemin = os.path.join(dossier, fichier)
        try:
            print(f"  [{i}/{len(fichiers_pdf)}] Chargement: {fichier}...", end=" ")
            chargeur_pdf = PyPDFLoader(chemin)
            docs_loaded = chargeur_pdf.load()

            # Appliquer le nettoyage à chaque document
            for doc in docs_loaded:
                doc.page_content = nettoyer_texte(doc.page_content)

            documents.extend(docs_loaded)
            print(f"✓ ({len(docs_loaded)} pages)")

        except KeyboardInterrupt:
            print("\n Chargement interrompu par l'utilisateur")
            raise
        except Exception as e:
            print(f" ERREUR: {type(e).__name__}")
            print(f"     Fichier ignoré: {fichier}")
            continue

    if not documents:
        raise FileNotFoundError(f"Aucun document valide trouvé dans {dossier}.")

    print(f"\n Total: {len(documents)} document(s) chargé(s).")
    return documents


def decouper_documents(documents: list) -> list:
    """Découpe les documents en chunks avec chevauchement"""
    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=TAILLE_CHUNK,
        chunk_overlap=CHEVAUCHEMENT_CHUNK,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    morceaux = decoupeur.split_documents(documents)
    print(f"{len(morceaux)} morceau(x) créé(s).")
    return morceaux


def creer_base_vecteurs(morceaux: list, dossier_persistance: str) -> Chroma:
    """Crée la base vectorielle ChromaDB par batch"""
    modele_embedding = creer_modele_embedding()

    print(f"\n Création de la base vectorielle avec {len(morceaux)} chunks")
    print(f"   Chargement par batch de 10")

    taille_batch = 10
    base = None

    for i in range(0, len(morceaux), taille_batch):
        batch_morceaux = morceaux[i : i + taille_batch]
        batch_num = i // taille_batch + 1
        total_batches = (len(morceaux) + taille_batch - 1) // taille_batch

        try:
            print(
                f"\n  [{batch_num}/{total_batches}] Traitement du batch {i}-{min(i+taille_batch, len(morceaux))}...",
                end=" ",
            )

            if base is None:
                base = Chroma.from_documents(
                    documents=batch_morceaux,
                    embedding=modele_embedding,
                    persist_directory=dossier_persistance,
                    collection_metadata={"hnsw:space": "cosine"},
                )
                print(f" Base créée ({len(batch_morceaux)} chunks)")
            else:
                # Retry logic pour éviter les erreurs de compaction ChromaDB
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        base.add_documents(batch_morceaux)
                        count = base._collection.count()
                        print(f" {len(batch_morceaux)} chunks ajoutés (Total: {count})")
                        break
                    except Exception as retry_err:
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt  # en séconde
                            print(
                                f"\n  Tentative {attempt + 1}/{max_retries} échouée, attente {wait_time}s..."
                            )
                            time.sleep(wait_time)
                        else:
                            raise

        except Exception as e:
            print(f"\n ERREUR lors du batch {batch_num}: {type(e).__name__}")
            print(f"   Message: {str(e)}")
            if base:
                print(
                    f"   Chunks actuellement dans la base: {base._collection.count()}"
                )
            raise

    total_count = base._collection.count() if base else 0
    print(f"\n Base vectorielle créée dans {dossier_persistance}.")
    print(f"   Total chunks indexés: {total_count}/{len(morceaux)}")
    return base


def charger_base_existante(dossier_persistance: str) -> Chroma:
    """Charge une base vectorielle existante"""
    modele_embedding = creer_modele_embedding()
    base = Chroma(
        persist_directory=dossier_persistance,
        embedding_function=modele_embedding,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Base vectorielle existante chargée ({base._collection.count()} entrées).")
    return base


def sauvegarder_documents_bruts(documents: list, dossier_sortie: str):
    """Sauvegarde le texte extrait des PDFs par fichier source"""
    os.makedirs(dossier_sortie, exist_ok=True)

    # Grouper les documents par source 
    documents_par_source = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in documents_par_source:
            documents_par_source[source] = []
        documents_par_source[source].append(doc.page_content)

    for source, contenu in documents_par_source.items():
        nom_fichier = os.path.basename(source).replace(".pdf", "_extracted.txt")
        chemin_sortie = os.path.join(dossier_sortie, nom_fichier)

        with open(chemin_sortie, "w", encoding="utf-8") as f:
            f.write(f"Source: {source}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("\n---PAGE---\n".join(contenu))

        print(f"Texte brut sauvegardé: {chemin_sortie}")


def sauvegarder_chunks(morceaux: list, dossier_sortie: str):
    """Sauvegarde les chunks extraits avec leurs métadonnées"""
    os.makedirs(dossier_sortie, exist_ok=True)

    # Sauvegarder les chunks individuels
    for i, chunk in enumerate(morceaux):
        nom_fichier = f"chunk_{i:06d}.txt"
        chemin_sortie = os.path.join(dossier_sortie, nom_fichier)

        with open(chemin_sortie, "w", encoding="utf-8") as f:
            # En-tête avec métadonnées
            f.write(f"Chunk ID: {i}\n")
            f.write(f"Source: {chunk.metadata.get('source', 'unknown')}\n")
            f.write(f"Page: {chunk.metadata.get('page', 'unknown')}\n")
            f.write("=" * 80 + "\n\n")
            # Contenu
            f.write(chunk.page_content)

    # Sauvegarder un JSON avec l'index et métadonnées
    index_file = os.path.join(dossier_sortie, "chunks_index.json")
    index_data = {
        "total_chunks": len(morceaux),
        "creation_date": datetime.now().isoformat(),
        "chunks_metadata": [
            {
                "id": i,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "unknown"),
                "size": len(chunk.page_content),
            }
            for i, chunk in enumerate(morceaux)
        ],
    }

    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    # Sauvegarder tous les chunks dans un seul fichier JSON complet
    complete_file = os.path.join(dossier_sortie, "chunks_complete.json")
    complete_data = {
        "total_chunks": len(morceaux),
        "creation_date": datetime.now().isoformat(),
        "chunks": [
            {
                "id": i,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", "unknown"),
                "size": len(chunk.page_content),
                "content": chunk.page_content,
            }
            for i, chunk in enumerate(morceaux)
        ],
    }

    with open(complete_file, "w", encoding="utf-8") as f:
        json.dump(complete_data, f, indent=2, ensure_ascii=False)

    print(f" {len(morceaux)} chunks sauvegardés dans {dossier_sortie}/")
    print(f" Index JSON créé: {index_file}")
    print(f" Fichier complet JSON créé: {complete_file}")


def executer_ingestion() -> Chroma:
    # Point d'entrée principal il ingère les catalogues ou charge la base existante
    if os.path.exists(DOSSIER_BASE_VECTEURS) and os.listdir(DOSSIER_BASE_VECTEURS):
        print("Base vectorielle existante détectée. Chargement en cours...")
        return charger_base_existante(DOSSIER_BASE_VECTEURS)

    print("Première exécution : création de la base vectorielle...")

    # Étape 1: Charger les documents
    documents = charger_documents(DOSSIER_DONNEES_BRUTES)

    # Étape 2: Sauvegarder le texte brut extrait
    print("\nSauvegarde du texte brut extrait...")
    sauvegarder_documents_bruts(documents, DOSSIER_DONNEES_TRAITEES)

    # Étape 3: Découper en chunks
    morceaux = decouper_documents(documents)

    # Étape 4: Sauvegarder les chunks
    print("\nSauvegarde des chunks extraits...")
    dossier_chunks = os.path.join(DOSSIER_DONNEES_TRAITEES, "chunks")
    sauvegarder_chunks(morceaux, dossier_chunks)

    # Étape 5: Créer la base vectorielle
    print("\nCréation des vecteurs d'embedding...")
    base = creer_base_vecteurs(morceaux, DOSSIER_BASE_VECTEURS)

    print("\nIngestion terminée.")
    return base


if __name__ == "__main__":
    executer_ingestion()
