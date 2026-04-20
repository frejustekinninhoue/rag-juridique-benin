---
title: RAG Juridique Benin
emoji: ⚖️
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
---



# ⚖️ Assistant Juridique Béninois — RAG

> Posez vos questions sur le **Droit du Travail** et le **Code Foncier** du Bénin.  
> Propulsé par Groq (LLaMA 3) + ChromaDB + Streamlit — **100% gratuit**

---

##  Architecture

```
Question utilisateur
        │
        ▼
  [Streamlit UI]
        │
        ▼
  [ChromaDB] ──── Embeddings (sentence-transformers)
        │
        ▼
  Top-5 chunks juridiques
        │
        ▼
  [Groq / LLaMA 3] ──── Prompt contextualisé
        │
        ▼
  Réponse + Sources citées
```

---

##  Installation locale (étape par étape)

### Étape 1 — Cloner et installer

```bash
git clone https://github.com/VOTRE_USERNAME/rag-juridique-benin.git
cd rag-juridique-benin

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Étape 2 — Obtenir une clé Groq gratuite

1. Allez sur https://console.groq.com
2. Créez un compte (gratuit)
3. Allez dans **API Keys** → **Create API Key**
4. Copiez la clé

### Étape 3 — Configurer l'environnement

```bash
cp .env.example .env
```

Ouvrez `.env` et remplacez `votre_cle_groq_ici` par votre vraie clé :
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### Étape 4 — Obtenir les textes de loi (PDFs)

Téléchargez les textes officiels et placez-les dans `data/raw/` :

| Document | Source | Nom de fichier recommandé |
|----------|--------|---------------------------|
| Code du Travail du Bénin (Loi N°98-004) | [Secrétariat général du Gouvernement du Bénin](https://sgg.gouv.bj/doc/loi-98-004) | `droit_travail_benin.pdf` |
| Code Foncier et Domanial (Loi N°2013-01) | [ANDF](https://andf.bj/wp-content/uploads/2024/10) | `code_foncier_benin.pdf` |
| Code Foncier et Domanial (Loi N°2017-15) | [Food and Agriculture Organization](https://faolex.fao.org/docs/pdf/Ben174252) | `loi_2017_15_modification_code_foncier.pdf` |



### Étape 5 — Indexer les documents

```bash
python -m rag.ingest
```

Durée : ~2-5 minutes selon la taille des PDFs.  
Résultat : la base vectorielle est créée dans `data/chroma_db/`

### Étape 6 — Lancer l'application

```bash
streamlit run app.py
```

Ouvrez http://localhost:8501 dans votre navigateur. 

---

##  Déploiement gratuit sur Streamlit Cloud

> Accessible à tout le monde, sans paiement.

### Étape 1 — Préparer la base vectorielle pour le déploiement

La base vectorielle doit être incluse dans le repo car Streamlit Cloud
ne peut pas la reconstruire (pas d'accès local aux PDFs).

```bash
# Après avoir indexé localement :
git add data/chroma_db/
# Retirez data/chroma_db/ de .gitignore temporairement
git commit -m "Add vector database"
git push
```

### Étape 2 — Créer un compte Streamlit Cloud

1. Allez sur https://streamlit.io/cloud
2. Connectez-vous avec votre compte GitHub (gratuit)

### Étape 3 — Déployer

1. Cliquez **New app**
2. Sélectionnez votre repo GitHub
3. Branch : `main`
4. Main file : `app.py`
5. Cliquez **Deploy**

### Étape 4 — Configurer les secrets

Dans Streamlit Cloud → **Settings** → **Secrets**, ajoutez :

```toml
GROQ_API_KEY = "gsk_votre_cle_ici"
GROQ_MODEL = "llama-3.1-8b-instant"
CHROMA_DB_PATH = "./data/chroma_db"
TOP_K = "8"
```

### Étape 5 — Votre app est en ligne ! 

Partagez le lien `https://VOTRE_APP.streamlit.app` avec tout le monde.

---

##  Structure du projet

```
rag-juridique-benin/
├── app.py                    # Interface Streamlit
├── requirements.txt          # Dépendances Python
├── .env.example              # Template des variables d'environnement
├── .gitignore
├── rag/
│   ├── __init__.py
│   ├── ingest.py             # Ingestion et indexation des PDFs
│   └── chain.py              # Chaîne RAG (retrieval + génération)
├── data/
│   ├── raw/                  # PDFs sources (non versionnés)
│   └── chroma_db/            # Base vectorielle (versionnée pour déploiement)
└── .streamlit/
    ├── config.toml           # Thème (couleurs du Bénin )
    └── secrets.toml.example  # Template secrets
```

---

##  Coûts — 100% gratuit

| Service | Plan gratuit |
|---------|-------------|
| **Groq API** | ~14 000 req/jour, 30 req/min |
| **Streamlit Cloud** | 1 app publique illimitée |
| **ChromaDB** | Local, aucun coût |
| **sentence-transformers** | Open source, local |

---

##  Personnalisation

**Ajouter d'autres textes de loi :**
```bash
# Placez le nouveau PDF dans data/raw/
# Relancez l'indexation (les nouveaux chunks s'ajoutent)
python -m rag.ingest
```

**Changer le modèle LLM :**
```
# Dans .env, remplacez par un autre modèle Groq gratuit :
GROQ_MODEL=llama-3.1-70b-versatile   # Plus puissant mais plus lent
GROQ_MODEL=mixtral-8x7b-32768        # Très bon pour le français
```

---

Application Streamlit déployée via Docker sur Hugging Face Spaces.

## Variables/Secrets à configurer dans HF Spaces
- `GROQ_API_KEY`
- `CHROMA_DB_PATH` (ex: `./data/chroma_db`)
- `GROQ_MODEL` (optionnel)


## ⚠️ Avertissement légal

Cet outil fournit des informations juridiques à titre **indicatif uniquement**.
Il ne remplace pas les conseils d'un professionnel du droit.
Pour tout litige, consultez un **avocat ou notaire agréé au Bénin**.
