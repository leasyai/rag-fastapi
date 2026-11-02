# PDF Question Answering with RAG and FastAPI

A personal project exploring **retrieval-augmented generation (RAG)** for question answering over PDF documents. The application combines PDF text processing, embeddings, vector similarity search, and a locally hosted language model behind a Python API.

I built this project to gain practical experience connecting NLP components into an application, and to work with testing, containerization, and automated build workflows rather than only experimenting in notebooks.

## What the project does

The intended workflow is:

1. Read text from a PDF document.
2. Split the text into smaller chunks.
3. Create vector embeddings for the chunks.
4. Store and retrieve relevant chunks using PostgreSQL and pgvector.
5. Use a locally running language model to generate an answer from retrieved context.

The repository also contains code for an in-memory vector store and similarity search experiments.

## Technologies

- **Python** and **FastAPI** for application endpoints
- **LlamaIndex** for document processing, chunking, and retrieval components
- **Hugging Face embeddings** for semantic representations
- **PostgreSQL / pgvector** for vector storage and similarity search
- **llama.cpp / LlamaCPP** for local language-model inference
- **Docker** for packaging the API
- **pytest** for automated tests
- **GitHub Actions** for CI and Docker image publishing to GitHub Container Registry (GHCR)

## Architecture

```text
PDF document
    |
    v
PDF extraction and text chunking
    |
    v
Embedding generation
    |
    v
PostgreSQL + pgvector
    |
    v
Similarity retrieval
    |
    v
Retrieved context + question
    |
    v
Local LLM server
    |
    v
Answer via FastAPI
```

The local LLM server and the API can run as separate processes. In my development setup, the LLM runs locally on a Mac and the API can be run in a Docker container. The LLM itself is not bundled into the Docker image.

## Repository structure

| File | Purpose |
| --- | --- |
| `main.py` | PDF processing, embeddings, vector search, and database-related logic |
| `rag_api.py` | RAG API implementation |
| `llm_server.py` | Local LLM inference endpoint |
| `test_main.py` | Tests for application components |
| `test_rag_api.py` | Tests for RAG API components |
| `Dockerfile` | Container definition |
| `requirements.txt` | Python dependencies |
| `requirements-rag.txt` | Additional RAG dependencies |
| `.github/workflows/ci.yml` | GitHub Actions workflow |

## Running locally

This repository is a **development project**, not a one-command hosted demo. It requires a local PostgreSQL database with the `vector` extension, an appropriate Python environment, and a compatible local language model.

### 1. Clone the repository

```bash
git clone https://github.com/leasyai/rag-fastapi.git
cd rag-fastapi
```

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Depending on which components you run, dependencies from `requirements-rag.txt` may also be needed:

```bash
python -m pip install -r requirements-rag.txt
```

### 3. Configure the local services

- Start PostgreSQL and enable the `vector` extension for the database used by the application.
- Configure the database connection values in the application for your own environment. **Do not commit real credentials.**
- Set up the local LLM and check the model path, available memory, and hardware acceleration settings in `llm_server.py`.
- If running the API in Docker while the LLM or database runs on the host, adjust hostnames and networking accordingly; `localhost` inside a container refers to the container itself.

> **Setup note:** Database initialization, environment-specific connection settings, and the complete end-to-end startup sequence are not yet automated in this repository. These steps need to be adapted to the machine where the application runs.

### 4. Run the tests

```bash
python -m pytest -q
```

### 5. Run the local LLM server

Once the local model and dependencies are configured, the LLM endpoint defined in `llm_server.py` can be started with:

```bash
uvicorn llm_server:app --host 127.0.0.1 --port 8001
```

The server exposes a `POST /generate` endpoint accepting a JSON body with a `prompt` field. Model initialization may take time and may download model files depending on the local configuration.

The RAG API has its own configuration and service requirements; consult `rag_api.py` before starting it. The commands above are a starting point, **not a verified end-to-end installation guide**.

## Engineering practices

- Unit/API test files are included in the repository.
- A GitHub Actions workflow is configured for CI and publishing a Docker image to GHCR.
- A Dockerfile is provided for the application environment.
- Dependencies are listed in requirements files.

These practices help make the project easier to inspect and reproduce, but they do not mean the application has been deployed or validated as a production service.

## Current limitations and possible improvements

- No public hosted demo or production deployment
- Local model inference requires sufficient hardware resources
- Database and service configuration is still environment-specific
- No documented evaluation benchmark for retrieval quality or answer faithfulness
- Production concerns such as authentication, observability, scaling, and robust error handling would require further work

Possible next steps include simplifying local setup, adding sample test documents, measuring retrieval and answer quality, and improving configuration management.

## Project status

**Personal learning and engineering project.** The goal is to demonstrate practical experience building and testing an LLM-based application, not to present it as a production-ready service.
