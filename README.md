# LLM Twin - Content Generation based on your Unique styling using LLMs

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [System Components](#-system-components)
- [Project Structure](#-project-structure)
- [Installation](#-installation)

---

## 🎯 Overview

**LLM Twin** is a comprehensive LLMOps system that creates an AI-powered "digital twin" by:

1. **Crawling** digital content (Medium articles, LinkedIn posts, GitHub repositories, Twitter/X posts)
2. **Processing** and cleaning the data through feature engineering pipelines
3. **Embedding** content into vector representations for semantic search
4. **Retrieving** relevant context using advanced RAG techniques
5. **Generating** personalized content using fine-tuned LLMs/Gemini

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Ingestion Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Crawlers: Medium │ LinkedIn │ GitHub │ Twitter/X │ Custom      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer (MongoDB)                     │
│         Raw Documents: Posts │ Articles │ Repositories          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Feature Engineering Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Data Cleaning    │  2. Chunking    │  3. Embedding          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Storage (Qdrant)                       │
│         Embedded Chunks with Semantic Search Index               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG Retrieval System                       │
├─────────────────────────────────────────────────────────────────┤
│  Query Expansion │ Self-Query │ Semantic Search │ Reranking     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Inference (Gemini)                        │
│              Content Generation with Context                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 System Components

### 1. **Data Crawlers** (`llm_engineering/application/crawlers/`)

Specialized crawlers for different platforms:

- **MediumCrawler**: Extracts articles using Selenium
- **LinkedInCrawler**: Scrapes posts and profiles (deprecated due to LinkedIn changes)
- **GithubCrawler**: Clones repositories and extracts code
- **TwitterCrawler**: Fetches tweets via ScrapingDog API
- **CustomArticleCrawler**: Generic HTML content extraction

### 2. **Data Storage**

**MongoDB (NoSQL)**
- Stores raw documents with metadata
- Collections: `posts`, `articles`, `repositories`, `users`

**Qdrant (Vector DB)**
- Stores embedded chunks for similarity search
- Collections: `embedded_posts`, `embedded_articles`, `embedded_repositories`

### 3. **Feature Engineering** (`llm_engineering/application/preprocessing/`)

**Cleaning Pipeline**
- Removes special characters and normalizes text
- Extracts meaningful content from structured data

**Chunking Pipeline**
- Posts: 250 tokens (overlap: 25)
- Articles: 1000-2000 characters (semantic chunking)
- Repositories: 1500 tokens (overlap: 100)

**Embedding Pipeline**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Generates 384-dimensional embeddings
- Batch processing for efficiency

### 4. **RAG System** (`llm_engineering/application/rag/`)

**Query Expansion**
- Generates multiple query variations using Gemini
- Improves recall by exploring different perspectives

**Self-Query**
- Extracts author information from queries
- Filters results by user context

**Retrieval**
- Parallel search across data categories
- Top-k selection with diversity

**Reranking**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-4-v2`
- Refines results based on query-document relevance

### 5. **LLM Inference** (`llm_engineering/model/inference/`)

- **Model**: Google Gemini 2.0 Flash
- **Parameters**: Temperature=0.0, Max tokens=256
- **Prompt Engineering**: Context-aware generation
- **Monitoring**: Opik/Comet ML integration

---

## 📁 Project Structure

```
llm-twin/
├── llm_engineering/              # Core application code
│   ├── application/               # Application layer
│   │   ├── crawlers/              # Data extraction
│   │   │   ├── base.py            # Abstract crawler classes
│   │   │   ├── dispatcher.py     # Crawler routing
│   │   │   ├── medium.py          # Medium scraper
│   │   │   ├── linkedin.py        # LinkedIn scraper
│   │   │   ├── github.py          # GitHub scraper
│   │   │   ├── twitter.py         # Twitter/X scraper
│   │   │   └── custom_article.py  # Generic article scraper
│   │   │
│   │   ├── networks/              # ML models
│   │   │   ├── base.py            # Singleton pattern
│   │   │   └── embeddings.py      # Embedding & reranking models
│   │   │
│   │   ├── preprocessing/         # Data transformation
│   │   │   ├── cleaning_data_handlers.py
│   │   │   ├── chunking_data_handlers.py
│   │   │   ├── embedding_data_handlers.py
│   │   │   ├── dispatchers.py     # Processing orchestration
│   │   │   └── operations/        # Core operations
│   │   │       ├── cleaning.py
│   │   │       └── chunking.py
│   │   │
│   │   ├── rag/                   # Retrieval system
│   │   │   ├── base.py
│   │   │   ├── query_expansion.py
│   │   │   ├── self_query.py
│   │   │   ├── retriever.py       # Main retrieval logic
│   │   │   ├── reranking.py
│   │   │   └── prompt_templates.py
│   │   │
│   │   └── utils/                 # Utilities
│   │       ├── misc.py
│   │       └── split_user_full_name.py
│   │
│   ├── domain/                    # Domain models
│   │   ├── base/                  # Base classes
│   │   │   ├── nosql.py           # MongoDB ODM
│   │   │   └── vector.py          # Qdrant ODM
│   │   ├── documents.py           # Raw document models
│   │   ├── cleaned_documents.py   # Cleaned data models
│   │   ├── chunks.py              # Chunk models
│   │   ├── embedded_chunks.py     # Embedded chunk models
│   │   ├── queries.py             # Query models
│   │   ├── types.py               # Enums & types
│   │   └── exceptions.py          # Custom exceptions
│   │
│   ├── infrastructure/            # External integrations
│   │   ├── db/
│   │   │   ├── mongo.py           # MongoDB connection
│   │   │   └── qdrant.py          # Qdrant connection
│   │   ├── opik_utils.py          # Monitoring setup
│   │   └── inference_pipeline_api.py  # FastAPI service
│   │
│   ├── model/                     # LLM inference
│   │   └── inference/
│   │       ├── inference.py       # Gemini integration
│   │       ├── run.py             # Execution logic
│   │       └── test.py            # Manual testing
│   │
│   └── settings.py                # Configuration management
│
├── pipelines/                     # ZenML pipelines
│   ├── digital_data_etl.py        # Data ingestion pipeline
│   └── feature_engineering.py     # Processing pipeline
│
├── steps/                         # Pipeline steps
│   ├── etl/
│   │   ├── get_or_create_user.py
│   │   └── crawl_links.py
│   └── feature_engineering/
│       ├── query_data_warehouse.py
│       ├── clean.py
│       ├── rag.py                 # Chunk & embed
│       └── load_to_vector_db.py
│
├── tools/
│   └── run.py                     # CLI entry point
│
├── configs/                       # Pipeline configurations
│   ├── digital_data_etl_*.yaml
│   └── feature_engineering.yaml
│
├── pyproject.toml                 # Dependencies & tasks
├── .python-version                # Python version (3.11)
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11
- [uv](https://github.com/astral-sh/uv) package manager
- MongoDB instance (local or Atlas)
- Qdrant instance (local or cloud)
- Google API key (Gemini)
- Optional: ScrapingDog API key (for Twitter), Comet ML (monitoring)

### Setup Steps

1. **Install uv package manager** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone https://github.com/yourusername/llm-twin.git
cd llm-twin
```

3. **Install dependencies**:
```bash
uv sync
```

This reads `pyproject.toml` and installs all required packages in a virtual environment.

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start required services**:

**MongoDB** (Docker):
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

**Qdrant** (Docker):
```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest
```

**ZenML** (optional, for UI):
```bash
uv run zenml up
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Database
DATABASE_NAME=llm-twin
DATABASE_HOST=mongodb://localhost:27017

# Qdrant
USE_QDRANT_CLOUD=false
QDRANT_DATABASE_HOST=localhost
QDRANT_DATABASE_PORT=6333
# For cloud:
# USE_QDRANT_CLOUD=true
# QDRANT_CLOUD_URL=https://your-cluster.qdrant.io
# QDRANT_APIKEY=your-api-key

# Google Gemini
GOOGLE_API_KEY=your-google-api-key
GOOGLE_GEMINI_MODEL=gemini-2.0-flash

# Models
TEXT_EMBEDDING_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2
RERANKING_CROSS_ENCODER_MODEL_ID=cross-encoder/ms-marco-MiniLM-L-4-v2
RAG_MODEL_DEVICE=cpu

# Inference
TEMPERATURE_INFERENCE=0.0
MAX_NEW_TOKENS_INFERENCE=256
TOP_P_INFERENCE=0.9

# Optional: Twitter scraping
SCRAPINGDOG_API_KEY=your-scrapingdog-key

# Optional: Monitoring
COMET_API_KEY=your-comet-key
COMET_PROJECT=llm-twin

# Optional: HuggingFace
HF_TOKEN=your-hf-token
HF_USERNAME=your-username
```

### Pipeline Configurations (`configs/`)

**ETL Config Example** (`digital_data_etl_paul_iusztin.yaml`):
```yaml
parameters:
  user_full_name: "Paul Iusztin"
  links:
    - "https://medium.com/@pauliusztin/article-slug"
    - "https://github.com/username/repo"
    - "https://x.com/username/status/123456789"
```

**Feature Engineering Config** (`feature_engineering.yaml`):
```yaml
parameters:
  author_full_names:
    - "Paul Iusztin"
    - "Maxime Labonne"
```

---

## 💻 Usage

### 1. Data Ingestion (ETL Pipeline)

Extract content from digital platforms:

```bash
# Run ETL for a specific user
uv run poe run-digital-data-etl-paul

# Or run for multiple users
uv run poe run-digital-data-etl

# With custom config
uv run python -m tools.run --run-etl --etl-config-filename your_config.yaml
```

### 2. Feature Engineering Pipeline

Process raw data into embeddings:

```bash
# Run feature engineering
uv run poe run-feature-engineering-pipeline

# Or directly
uv run python -m tools.run --run-feature-engineering
```


### 3. API Service (FastAPI)

Start the inference API:

```bash
uv run uvicorn llm_engineering.infrastructure.inference_pipeline_api:app --reload --port 8000
```

**Query the API:**

```bash
curl -X POST "http://localhost:8000/rag" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain vector databases"}'
```

**Response:**
```json
{
  "answer": "Vector databases are specialized systems designed for..."
}
```

