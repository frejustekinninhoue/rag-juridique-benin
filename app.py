"""
app.py
─────────────────────────────────────────────────────────────────────────────
Interface Streamlit du RAG Juridique Béninois
+ Architecture Chat Hybride :
  1) Mode Juridique Bénin (base Chroma locale)
  2) Mode Document Upload (chat with your document)
  3) Mode Hybride (Juridique + Document)
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# LangChain / RAG
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Parsers documents
import pdfplumber

try:
    import docx  # python-docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

load_dotenv()

# ── Configuration globale ────────────────────────────────────────────────────
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
UPLOAD_COLLECTION_PREFIX = "uploaded_doc_"

# ── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="⚖️ Assistant Juridique Béninois",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.write("DEBUG cwd:", os.getcwd())
# st.write("DEBUG CHROMA_DB_PATH env:", os.getenv("CHROMA_DB_PATH"))
# st.write("DEBUG CHROMA_DB_PATH exists:", Path(os.getenv("CHROMA_DB_PATH", "./data/chroma_db")).exists())
# st.write("DEBUG ./data exists:", Path("./data").exists())
# st.write("DEBUG ./data/chroma_db exists:", Path("./data/chroma_db").exists())

# ── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #006B3F 0%, #FCD116 50%, #E8112D 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 24px;
    }
    .main-header h1 { color: white; font-size: 2rem; margin: 0; text-shadow: 1px 1px 3px #000; }
    .main-header p  { color: #f0f0f0; margin: 4px 0 0 0; font-size: 1rem; }

    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #006B3F;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.85rem;
    }

    .source-card-upload {
        background: #eef7ff;
        border-left: 4px solid #0d6efd;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 0.85rem;
    }

    .domain-badge {
        background: #006B3F;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }

    .upload-badge {
        background: #0d6efd;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
    }

    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        margin: 12px 0;
        font-size: 0.85rem;
    }

    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_collection_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", stem)
    return f"{UPLOAD_COLLECTION_PREFIX}{stem}"[:63]


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    pages = []
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        with pdfplumber.open(temp_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = _clean_text(text)
                if len(text) > 40:
                    pages.append({
                        "text": text,
                        "page": i,
                    })
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return pages


def _extract_text_from_txt(file_bytes: bytes) -> List[Dict[str, Any]]:
    text = file_bytes.decode("utf-8", errors="ignore")
    text = _clean_text(text)
    if not text:
        return []
    return [{"text": text, "page": 1}]


def _extract_text_from_docx(file_bytes: bytes) -> List[Dict[str, Any]]:
    if not DOCX_AVAILABLE:
        return []

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        document = docx.Document(temp_path)
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
        text = _clean_text("\n".join(paragraphs))
        if not text:
            return []
        return [{"text": text, "page": 1}]
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def _chunk_records(records: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", "Article", "ARTICLE", ". ", " "],
    )

    chunks = []
    for rec in records:
        parts = splitter.split_text(rec["text"])
        for idx, part in enumerate(parts):
            p = part.strip()
            if len(p) > 80:
                chunks.append({
                    "text": p,
                    "metadata": {
                        "source": source_name,
                        "page": rec.get("page", 1),
                        "domain": "Document Utilisateur",
                        "chunk_idx": idx,
                    },
                })
    return chunks


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def index_uploaded_file(uploaded_file) -> str:
    file_name = uploaded_file.name
    suffix = Path(file_name).suffix.lower()
    file_bytes = uploaded_file.getvalue()

    if suffix == ".pdf":
        records = _extract_text_from_pdf(file_bytes)
    elif suffix == ".txt":
        records = _extract_text_from_txt(file_bytes)
    elif suffix == ".docx":
        records = _extract_text_from_docx(file_bytes)
    else:
        records = []

    if not records:
        raise ValueError("Impossible d'extraire du texte exploitable depuis ce document.")

    chunks = _chunk_records(records, source_name=file_name)
    if not chunks:
        raise ValueError("Le document est trop court ou illisible après extraction.")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection_name = _safe_collection_name(file_name)
    embeddings = get_embeddings()

    # Rebuild collection for same file name
    vs = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    try:
        vs.delete_collection()
    except Exception:
        pass

    Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DB_PATH,
        collection_name=collection_name,
    )
    return collection_name


def format_sources_ui(sources: List[Dict[str, Any]], uploaded: bool = False):
    for src in sources:
        if uploaded:
            st.markdown(f"""
            <div class='source-card-upload'>
                <span class='upload-badge'>Document Upload</span>
                &nbsp; <b>{src.get('fichier', 'Inconnu')}</b> — Page {src.get('page', '?')}<br>
                <i>{src.get('extrait', '')}</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='source-card'>
                <span class='domain-badge'>{src.get('domaine', '')}</span>
                &nbsp; <b>{src.get('fichier', 'Inconnu')}</b> — Page {src.get('page', '?')}<br>
                <i>{src.get('extrait', '')}</i>
            </div>
            """, unsafe_allow_html=True)


# ── Chargement des chaînes RAG ──────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Chargement du système juridique...")
def load_legal_rag():
    chroma_path = Path(CHROMA_DB_PATH)
    if not chroma_path.exists():
        return None, None, "no_db"

    try:
        from rag.chain import build_rag_chain
        chain_builder, retriever = build_rag_chain()
        return chain_builder, retriever, "ok"
    except ValueError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Erreur : {e}"


@st.cache_resource(show_spinner=False)
def load_upload_rag(collection_name: str):
    from rag.chain import build_uploaded_doc_rag_chain
    return build_uploaded_doc_rag_chain(collection_name=collection_name, persist_directory=CHROMA_DB_PATH)


@st.cache_resource(show_spinner=False)
def load_hybrid_rag(collection_name: str):
    from rag.chain import build_hybrid_rag_chain
    return build_hybrid_rag_chain(collection_name=collection_name, persist_directory=CHROMA_DB_PATH)


chain_builder_legal, retriever_legal, status = load_legal_rag()

# ── Gestion erreurs chargement base juridique ───────────────────────────────
if status == "no_db":
    st.warning("""
### ⚠️ Base juridique non trouvée
Le mode **Juridique Bénin** et **Hybride** nécessitent une base indexée.

Étapes :
1. Placez vos PDFs juridiques dans `data/raw/`
2. Lancez : `python -m rag.ingest`
""")
elif status != "ok":
    st.error(f"### ❌ Erreur de configuration\n\n{status}")
    st.info("Vérifiez votre fichier `.env` et votre clé `GROQ_API_KEY`.")

# ── En-tête ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ Assistant Juridique Béninois</h1>
    <p>Architecture Hybride : Juridique Bénin + Chat with your document</p>
</div>
""", unsafe_allow_html=True)

# ── État session ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Bonjour ! Je peux répondre en 3 modes :\n\n"
            "- ⚖️ **Juridique Bénin** : base de lois indexées\n"
            "- 📄 **Document Upload** : questions sur votre document\n"
            "- 🔀 **Hybride** : combinaison des deux\n\n"
            "Choisissez un mode dans la barre latérale puis posez votre question."
        ),
        "sources_legal": [],
        "sources_upload": [],
    }]

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = ""

if "uploaded_collection_name" not in st.session_state:
    st.session_state["uploaded_collection_name"] = None

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Flag_of_Benin.svg/320px-Flag_of_Benin.svg.png", width=100)

    st.markdown("### 🧠 Mode de chat")
    mode = st.radio(
        "Choisissez le mode",
        options=["Juridique Bénin", "Document Upload", "Hybride"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📄 Chat with your document")
    uploaded_file = st.file_uploader(
        "Importer un PDF, TXT ou DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=False,
        help="Le document sera indexé localement puis interrogé en langage naturel.",
    )

    if uploaded_file is not None:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Indexer le document", use_container_width=True):
                try:
                    with st.spinner("Indexation du document en cours..."):
                        collection_name = index_uploaded_file(uploaded_file)
                    st.session_state["uploaded_collection_name"] = collection_name
                    st.session_state["uploaded_filename"] = uploaded_file.name
                    st.success(f"Document indexé : {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Erreur d'indexation : {e}")

        with col_b:
            if st.button("Effacer document", use_container_width=True):
                st.session_state["uploaded_collection_name"] = None
                st.session_state["uploaded_filename"] = None
                st.info("Document upload effacé de la session.")

    if st.session_state["uploaded_filename"]:
        st.caption(f"Document actif : `{st.session_state['uploaded_filename']}`")

    st.markdown("---")
    st.markdown("### 📚 Domaines couverts")
    st.success("✅ Droit du Travail")
    st.success("✅ Code Foncier et Domanial")

    st.markdown("---")
    st.markdown("### 💡 Exemples de questions")
    example_questions = [
        "Quelle est la durée légale du travail au Bénin ?",
        "Comment se déroule un licenciement abusif ?",
        "Quels sont les droits d'un travailleur en cas de maladie ?",
        "Comment acquérir un terrain au Bénin légalement ?",
        "Quels sont les points clés du document que j'ai uploadé ?",
        "Compare la règle du Code du travail avec mon document.",
    ]
    for q in example_questions:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#666;'>
    ⚠️ Outil informatif uniquement.<br>
    Pour toute décision importante, consultez un <b>avocat agréé</b>.
    </div>
    """, unsafe_allow_html=True)

# ── Affichage historique ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        legal_sources = msg.get("sources_legal", [])
        upload_sources = msg.get("sources_upload", [])

        if legal_sources:
            with st.expander(f"📎 Sources juridiques ({len(legal_sources)})"):
                format_sources_ui(legal_sources, uploaded=False)

        if upload_sources:
            with st.expander(f"📎 Sources document upload ({len(upload_sources)})"):
                format_sources_ui(upload_sources, uploaded=True)

# ── Avertissement légal ──────────────────────────────────────────────────────
st.markdown("""
<div class='warning-box'>
⚠️ <b>Avertissement :</b> Cet assistant fournit des informations juridiques à titre indicatif.
Il ne remplace pas les conseils d'un professionnel du droit.
Pour tout litige ou décision importante, consultez un <b>avocat ou notaire agréé au Bénin</b>.
</div>
""", unsafe_allow_html=True)

# ── Entrée utilisateur ───────────────────────────────────────────────────────
default_input = st.session_state.pop("pending_question", "")
question = st.chat_input("Posez votre question...") or default_input

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources_legal": [],
        "sources_upload": [],
    })
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            if mode == "Juridique Bénin":
                if status != "ok" or not chain_builder_legal or not retriever_legal:
                    raise ValueError("Base juridique indisponible. Lancez `python -m rag.ingest` puis redémarrez l'app.")

                with st.spinner("🔍 Recherche dans les textes de loi..."):
                    from rag.chain import get_answer_with_sources
                    result = get_answer_with_sources(question, chain_builder_legal, retriever_legal)

                answer = result.get("answer", "")
                legal_sources = result.get("sources", [])
                upload_sources = []

            elif mode == "Document Upload":
                col_name = st.session_state.get("uploaded_collection_name")
                if not col_name:
                    raise ValueError("Aucun document upload indexé. Importez puis cliquez sur 'Indexer le document'.")

                with st.spinner("📄 Analyse du document upload..."):
                    chain_builder_upload, retriever_upload = load_upload_rag(col_name)
                    from rag.chain import get_answer_with_sources
                    result = get_answer_with_sources(question, chain_builder_upload, retriever_upload)

                answer = result.get("answer", "")
                legal_sources = []
                upload_sources = result.get("sources", [])

            else:  # Hybride
                if status != "ok" or not chain_builder_legal or not retriever_legal:
                    raise ValueError("Base juridique indisponible pour le mode hybride.")
                col_name = st.session_state.get("uploaded_collection_name")
                if not col_name:
                    raise ValueError("Le mode hybride nécessite aussi un document upload indexé.")

                with st.spinner("🔀 Fusion des connaissances (juridique + document)..."):
                    chain_builder_hybrid, retrievers = load_hybrid_rag(col_name)
                    from rag.chain import get_hybrid_answer_with_sources
                    result = get_hybrid_answer_with_sources(question, chain_builder_hybrid, retrievers)

                answer = result.get("answer", "")
                legal_sources = result.get("legal_sources", [])
                upload_sources = result.get("upload_sources", [])

            st.markdown(answer)

            if legal_sources:
                with st.expander(f"📎 Sources juridiques ({len(legal_sources)})"):
                    format_sources_ui(legal_sources, uploaded=False)

            if upload_sources:
                with st.expander(f"📎 Sources document upload ({len(upload_sources)})"):
                    format_sources_ui(upload_sources, uploaded=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources_legal": legal_sources,
                "sources_upload": upload_sources,
            })

        except Exception as e:
            err_msg = f"❌ Erreur : {e}"
            st.error(err_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err_msg,
                "sources_legal": [],
                "sources_upload": [],
            })

    st.rerun()
