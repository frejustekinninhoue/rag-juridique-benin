"""
rag/chain.py — Architecture Chat Hybride + Chat with your document
"""

from __future__ import annotations

import os
import re
import tempfile
import streamlit as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TOP_K = int(os.getenv("TOP_K", 8))

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_LEGAL = "droit_beninois"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# ═══════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

PROMPT_GENERAL = """Vous êtes l'Assistant Juridique officiel spécialisé dans le droit béninois.
Vous maîtrisez exclusivement :
- Le Code du Travail (Loi N°98-004 du 27 janvier 1998)
- La Loi N°2017-05 sur les conditions d'embauche
- Le Code Foncier et Domanial (Loi N°2013-01 modifiée par la Loi N°2017-15)

━━━━━━━━━━━━━━━━━━━━ RÈGLES ABSOLUES ━━━━━━━━━━━━━━━━━━━━
❌ INTERDIT : Inventer des articles, dates, montants ou procédures absents du contexte.
❌ INTERDIT : Répondre avec des connaissances générales hors du contexte fourni.
✅ OBLIGATOIRE : Citer chaque affirmation avec sa source exacte.
✅ OBLIGATOIRE : Si l'info est absente → dire "Les documents disponibles ne contiennent pas cette information."
✅ OBLIGATOIRE : Signaler si une procédure nécessite un notaire, un géomètre ou l'ANDF.

━━━━━━━━━━━━━━━━━━━━ STRUCTURE DE RÉPONSE ━━━━━━━━━━━━━━━━━━
**📌 Réponse directe :** [Une phrase résumant la réponse]

**📖 Analyse juridique :**
[Développement point par point avec citations d'articles]

**⚖️ Textes de référence :**
[Liste des articles cités : Art. X, Loi Y]

**⚠️ Points d'attention :**
[Nuances, exceptions, organismes à contacter]

━━━━━━━━━━━━━━━━━━━━ CONTEXTE JURIDIQUE ━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION : {question}

✍️ RÉPONSE (basée uniquement sur le contexte ci-dessus) :"""

PROMPT_TARIFS = """Vous êtes un expert en fiscalité et tarification foncière au Bénin.
Votre rôle UNIQUE est d'extraire les montants exacts présents dans les documents.

━━━━━━━━━━━━━━━━━━━━ RÈGLES STRICTES ━━━━━━━━━━━━━━━━━━━━━━
❌ INTERDIT ABSOLU : Estimer, arrondir ou inventer un tarif.
❌ INTERDIT : Utiliser des montants qui ne figurent PAS dans le contexte.
✅ Si le tarif n'est pas dans le contexte → dire EXACTEMENT :
   "Le montant exact ne figure pas dans les documents indexés.
    Contactez l'ANDF (Agence Nationale du Domaine et du Foncier)
    ou le Bureau Communal du Domaine et du Foncier de votre commune."
✅ Exprimer tous les montants en Francs CFA (FCFA).
✅ Distinguer : frais d'État / frais de commune / frais de géomètre.

━━━━━━━━━━━━━━━━━━━━ FORMAT DE RÉPONSE ━━━━━━━━━━━━━━━━━━━━━

| Type de frais | Montant (FCFA) | Base légale |
|---------------|----------------|-------------|
| ...           | ...            | Art. X      |

**📌 Notes importantes :**
- [Variable selon...] si applicable
- [Organisme compétent] pour le paiement

━━━━━━━━━━━━━━━━━━━━ CONTEXTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION SUR LES TARIFS : {question}

✍️ RÉPONSE (montants extraits du contexte uniquement) :"""

PROMPT_PROCEDURE = """Vous êtes un guide administratif expert des procédures juridiques au Bénin.
Votre spécialité : expliquer les étapes des procédures de manière chronologique et précise.

━━━━━━━━━━━━━━━━━━━━ RÈGLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ N'inventez aucune étape absente du contexte.
✅ Présentez les étapes dans l'ORDRE CHRONOLOGIQUE.
✅ Indiquez les DÉLAIS LÉGAUX mentionnés (ex: "dans un délai de 60 jours").
✅ Précisez l'ORGANISME COMPÉTENT pour chaque étape.
✅ Signalez les pièces justificatives requises si mentionnées.
✅ Citez l'article correspondant à chaque étape.

━━━━━━━━━━━━━━━━━━━━ FORMAT DE RÉPONSE ━━━━━━━━━━━━━━━━━━━━━

**📌 Résumé de la procédure :** [1 phrase]

**📋 Étapes chronologiques :**

**Étape 1 — [Nom de l'étape]** *(Art. X, Loi Y)*
→ Action : [Ce qu'il faut faire]
→ Où : [Organisme / lieu]
→ Délai : [Si mentionné dans le contexte]
→ Pièces : [Documents requis si mentionnés]

**Étape 2 — ...** *(Art. X)*
[etc.]

**⚠️ Points critiques :**
[Sanctions, conditions suspensives, cas particuliers]

━━━━━━━━━━━━━━━━━━━━ CONTEXTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION SUR LA PROCÉDURE : {question}

✍️ RÉPONSE (étapes issues du contexte uniquement) :"""

PROMPT_DOCUMENT = """Vous êtes un assistant d'analyse documentaire.
Répondez UNIQUEMENT à partir du contexte fourni du document utilisateur.
Si l'information n'est pas dans le contexte, dites clairement :
"Cette information n'apparaît pas dans le document fourni."

Soyez précis, structuré et citez les passages/source (fichier + page si disponible).

━━━━━━━━━━━━━━━━━━━━ CONTEXTE DOCUMENT ━━━━━━━━━━━━━━━━━━━━━
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION : {question}

✍️ RÉPONSE :"""

PROMPT_HYBRID = """Vous êtes un assistant hybride.
Vous devez combiner :
1) le contexte juridique béninois,
2) le contexte du document fourni par l'utilisateur.

Règles:
- Ne rien inventer.
- Si une info est absente des deux contextes, le dire explicitement.
- Distinguer ce qui vient des "Textes juridiques béninois" et du "Document utilisateur".
- Citer les sources pour chaque partie.

Format:
**📌 Réponse synthétique**
...
**⚖️ Ce que disent les textes juridiques béninois**
...
**📄 Ce que dit votre document**
...
**⚠️ Écarts / points à vérifier**
...

━━━━━━━━━━━━━━━━━━━━ CONTEXTE JURIDIQUE ━━━━━━━━━━━━━━━━━━━━
{legal_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━ CONTEXTE DOCUMENT ━━━━━━━━━━━━━━━━━━━━━
{doc_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ QUESTION : {question}

✍️ RÉPONSE :"""


# ═══════════════════════════════════════════════════════════════════════════
# ROUTAGE
# ═══════════════════════════════════════════════════════════════════════════

KEYWORDS_TARIFS = [
    "combien", "tarif", "prix", "frais", "coût", "montant",
    "payer", "paiement", "fcfa", "cfa", "taxe", "redevance",
    "budget", "euro", "financer", "financement",
]
KEYWORDS_PROCEDURE = [
    "comment", "étape", "procédure", "démarche", "obtenir",
    "constituer", "dossier", "délai", "comment faire", "comment obtenir",
    "comment acquérir", "enregistrer", "immatriculer", "déposer",
    "demande", "formulaire", "qui contacter", "où aller",
]


def route_prompt(question: str) -> ChatPromptTemplate:
    q = question.lower()
    if any(kw in q for kw in KEYWORDS_TARIFS):
        return ChatPromptTemplate.from_template(PROMPT_TARIFS)
    if any(kw in q for kw in KEYWORDS_PROCEDURE):
        return ChatPromptTemplate.from_template(PROMPT_PROCEDURE)
    return ChatPromptTemplate.from_template(PROMPT_GENERAL)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_embeddings():
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return emb


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key or api_key == "votre_cle_groq_ici":
        raise ValueError(
            "❌ Clé Groq manquante ! Ajoutez GROQ_API_KEY dans Streamlit Secrets."
        )
    return ChatGroq(api_key=api_key, model_name=GROQ_MODEL, temperature=0, max_tokens=2048)



def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"[^\x00-\x7Fàâäéèêëîïôùûüçœæ\s\-\.\,\;\:\!\?\(\)\[\]\"\'\/\%\°\n]", "", text)
    return text.strip()


def _extract_pdf_text(file_path: str, source_name: str) -> List[Dict[str, Any]]:
    docs = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and len(text.strip()) > 30:
                docs.append({
                    "text": clean_text(text),
                    "page": page_num,
                    "source": source_name,
                    "domain": "Document Utilisateur",
                })
    return docs


def _extract_text_like_bytes(raw_bytes: bytes, source_name: str) -> List[Dict[str, Any]]:
    try:
        text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = str(raw_bytes)
    text = clean_text(text)
    if len(text) < 20:
        return []
    return [{
        "text": text,
        "page": 1,
        "source": source_name,
        "domain": "Document Utilisateur",
    }]


def _chunk_raw_docs(raw_docs: List[Dict[str, Any]]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "Article", "ARTICLE", ". ", " "],
    )
    chunks: List[Document] = []
    for d in raw_docs:
        parts = splitter.split_text(d["text"])
        for i, p in enumerate(parts):
            if len(p.strip()) > 80:
                chunks.append(
                    Document(
                        page_content=p.strip(),
                        metadata={
                            "source": d.get("source", "Inconnu"),
                            "page": d.get("page", "?"),
                            "domain": d.get("domain", ""),
                            "chunk_idx": i,
                        },
                    )
                )
    return chunks


def format_docs(docs: List[Document]) -> str:
    formatted: List[str] = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata or {}
        source = m.get("source", "Inconnu")
        page = m.get("page", "?")
        domain = m.get("domain", "")
        formatted.append(
            f"[Document {i} — {domain} | {source} | Page {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def _format_sources(source_docs: List[Document]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for doc in source_docs:
        m = doc.metadata or {}
        key = f"{m.get('source')}_{m.get('page')}_{m.get('chunk_idx')}"
        if key in seen:
            continue
        seen.add(key)
        excerpt = doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else "")
        out.append({
            "fichier": m.get("source", "Inconnu"),
            "page": m.get("page", "?"),
            "domaine": m.get("domain", ""),
            "extrait": excerpt,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RETRIEVERS / CHAINS
# ═══════════════════════════════════════════════════════════════════════════

def load_retriever(
    persist_directory: str = CHROMA_DB_PATH,
    collection_name: str = COLLECTION_LEGAL,
):
    embeddings = _get_embeddings()
    # debug temporaire
    print("DEBUG embeddings type:", type(embeddings))
    print("DEBUG embeddings attrs:", hasattr(embeddings, "embed_query"), hasattr(embeddings, "embed_documents"))

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
    )


def build_rag_chain():
    llm = _get_llm()
    retriever = load_retriever()

    def build_chain_for_question(question: str):
        prompt = route_prompt(question)
        return (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    return build_chain_for_question, retriever


def build_document_retriever_from_upload(
    file_name: str,
    file_bytes: bytes,
    session_id: str,
    base_tmp_dir: Optional[str] = None,
):
    suffix = Path(file_name).suffix.lower()
    safe_session = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)

    if base_tmp_dir is None:
        base_tmp_dir = tempfile.gettempdir()

    persist_dir = str(Path(base_tmp_dir) / f"rag_doc_{safe_session}")
    collection_name = f"user_doc_{safe_session}"

    # 1) extraction
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        raw_docs = _extract_pdf_text(tmp_path, source_name=file_name)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    else:
        # TXT / MD / fallback bytes
        raw_docs = _extract_text_like_bytes(file_bytes, source_name=file_name)

    # 2) chunk
    docs = _chunk_raw_docs(raw_docs)
    if not docs:
        raise ValueError("Impossible d'extraire du texte exploitable depuis le fichier fourni.")

    # 3) vectorstore
    embeddings = _get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": min(6, TOP_K), "fetch_k": min(24, TOP_K * 3)},
    )


def answer_legal_only(question: str, chain_builder, retriever) -> Dict[str, Any]:
    source_docs = retriever.invoke(question)
    chain = chain_builder(question)
    answer = chain.invoke(question)
    return {"answer": answer, "sources": _format_sources(source_docs)}


def answer_document_only(question: str, doc_retriever) -> Dict[str, Any]:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_DOCUMENT)

    source_docs = doc_retriever.invoke(question)

    chain = (
        {
            "context": doc_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return {"answer": answer, "sources": _format_sources(source_docs)}


def answer_hybrid(question: str, legal_retriever, doc_retriever) -> Dict[str, Any]:
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_HYBRID)

    legal_docs = legal_retriever.invoke(question)
    doc_docs = doc_retriever.invoke(question)

    legal_context = format_docs(legal_docs)
    doc_context = format_docs(doc_docs)

    chain = (
        {
            "legal_context": lambda _: legal_context,
            "doc_context": lambda _: doc_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)

    legal_sources = _format_sources(legal_docs)
    doc_sources = _format_sources(doc_docs)
    all_sources = legal_sources + doc_sources

    return {
        "answer": answer,
        "sources": all_sources,
        "sources_by_scope": {
            "juridique_benin": legal_sources,
            "document_utilisateur": doc_sources,
        },
    }


def build_uploaded_doc_rag_chain(
    collection_name: str,
    persist_directory: str = CHROMA_DB_PATH,
):
    """
    Construit une chaîne RAG dédiée à une collection de document uploadé.
    Retourne (chain_builder, retriever) pour compatibilité avec l'app Streamlit.
    """
    llm = _get_llm()
    retriever = load_retriever(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_DOCUMENT)

    def build_chain_for_question(_: str):
        return (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    return build_chain_for_question, retriever


def get_document_answer_with_sources(question: str, chain_builder, retriever) -> Dict[str, Any]:
    """
    Exécute une QA sur document uploadé avec sources.
    """
    source_docs = retriever.invoke(question)
    chain = chain_builder(question)
    answer = chain.invoke(question)
    return {"answer": answer, "sources": _format_sources(source_docs)}


def build_hybrid_rag_chain(
    collection_name: str,
    persist_directory: str = CHROMA_DB_PATH,
):
    """
    Construit un builder hybride (base juridique + document uploadé).
    Retourne (chain_builder, retrievers_dict) pour compatibilité avec l'app.
    """
    llm = _get_llm()
    legal_retriever = load_retriever(
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_LEGAL,
    )
    upload_retriever = load_retriever(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_HYBRID)

    def build_chain_for_question(question: str):
        legal_docs = legal_retriever.invoke(question)
        upload_docs = upload_retriever.invoke(question)
        legal_context = format_docs(legal_docs)
        doc_context = format_docs(upload_docs)

        return (
            {
                "legal_context": lambda _: legal_context,
                "doc_context": lambda _: doc_context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    retrievers = {
        "legal": legal_retriever,
        "upload": upload_retriever,
    }
    return build_chain_for_question, retrievers


def get_hybrid_answer_with_sources(question: str, chain_builder, retrievers) -> Dict[str, Any]:
    """
    Exécute une QA hybride et renvoie les sources séparées (juridique / upload).
    """
    legal_retriever = retrievers["legal"]
    upload_retriever = retrievers["upload"]

    legal_docs = legal_retriever.invoke(question)
    upload_docs = upload_retriever.invoke(question)

    chain = chain_builder(question)
    answer = chain.invoke(question)

    return {
        "answer": answer,
        "legal_sources": _format_sources(legal_docs),
        "upload_sources": _format_sources(upload_docs),
    }


# Compatibilité avec l'app existante
def get_answer_with_sources(question: str, chain_builder, retriever) -> Dict[str, Any]:
    return answer_legal_only(question, chain_builder, retriever)
