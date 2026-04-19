"""
rag/ingest.py
─────────────────────────────────────────────────────────────────────────────
Ingestion des documents juridiques béninois :
  - Lecture des PDFs (droit du travail + code foncier)
  - Découpage en chunks intelligents
  - Génération des embeddings
  - Stockage dans ChromaDB
"""

import os
import re
from pathlib import Path
from tqdm import tqdm

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
DATA_RAW_PATH  = Path("./data/raw")

# Modèle d'embeddings multilingue (français inclus) — 100% gratuit & local
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Taille des chunks : ~500 tokens avec chevauchement de 100
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """Nettoie le texte extrait du PDF."""
    # Supprime les lignes vides multiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Supprime les espaces multiples
    text = re.sub(r' {2,}', ' ', text)
    # Supprime les caractères parasites courants dans les PDFs
    text = re.sub(r'[^\x00-\x7Fàâäéèêëîïôùûüçœæ\s\-\.\,\;\:\!\?\(\)\[\]\"\'\/\%\°\n]', '', text)
    return text.strip()


def extract_pdf(pdf_path: Path) -> list[dict]:
    """
    Extrait le texte d'un PDF page par page.
    Retourne une liste de dicts {text, page, source}.
    """
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and len(text.strip()) > 50:  # ignore pages vides
                documents.append({
                    "text"   : clean_text(text),
                    "page"   : page_num,
                    "source" : pdf_path.name,
                    "domain" : detect_domain(pdf_path.name),
                })
    return documents


def detect_domain(filename: str) -> str:
    """Détecte le domaine juridique depuis le nom de fichier."""
    filename_lower = filename.lower()
    if any(kw in filename_lower for kw in ["travail", "labor", "emploi", "salaire"]):
        return "Droit du Travail"
    elif any(kw in filename_lower for kw in ["foncier", "terre", "cadastre", "propriete"]):
        return "Code Foncier"
    return "Juridique Général"


def chunk_documents(raw_docs: list[dict]) -> list[dict]:
    """
    Découpe les documents en chunks avec chevauchement.
    Préserve les métadonnées (source, page, domaine).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "Article", "ARTICLE", ". ", " "],
    )

    chunks = []
    for doc in raw_docs:
        parts = splitter.split_text(doc["text"])
        for i, part in enumerate(parts):
            if len(part.strip()) > 100:  # ignore les chunks trop courts
                chunks.append({
                    "text"     : part.strip(),
                    "metadata" : {
                        "source"    : doc["source"],
                        "page"      : doc["page"],
                        "domain"    : doc["domain"],
                        "chunk_idx" : i,
                    }
                })
    return chunks


def build_vectorstore(chunks: list[dict]) -> Chroma:
    """
    Génère les embeddings et stocke dans ChromaDB.
    """
    print(f"\n Chargement du modèle d'embeddings : {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    texts     = [c["text"]     for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print(f" Construction de la base vectorielle ({len(texts)} chunks)...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_PATH,
        collection_name="droit_beninois",
    )
    vectorstore.persist()
    print(f" Base vectorielle sauvegardée dans '{CHROMA_DB_PATH}'")
    return vectorstore


def ingest_all() -> None:
    """
    Point d'entrée principal :
    Lit tous les PDFs dans data/raw/ et construit la base vectorielle.
    """
    pdf_files = list(DATA_RAW_PATH.glob("*.pdf"))

    if not pdf_files:
        print(f"""
   Aucun PDF trouvé dans '{DATA_RAW_PATH}'.

  Placez vos PDFs dans ce dossier avec des noms explicites :
   - droit_travail_benin.pdf
   - code_foncier_benin.pdf
""")
        return

    print(f"\n {len(pdf_files)} PDF(s) trouvé(s) :")
    all_chunks = []

    for pdf_path in tqdm(pdf_files, desc="Traitement des PDFs"):
        print(f"\n   {pdf_path.name}")
        raw_docs = extract_pdf(pdf_path)
        print(f"     → {len(raw_docs)} pages extraites")
        chunks = chunk_documents(raw_docs)
        print(f"     → {len(chunks)} chunks générés")
        all_chunks.extend(chunks)

    print(f"\n Total : {len(all_chunks)} chunks à indexer")
    build_vectorstore(all_chunks)
    print("\n Ingestion terminée ! Vous pouvez lancer l'application.")


if __name__ == "__main__":
    ingest_all()
