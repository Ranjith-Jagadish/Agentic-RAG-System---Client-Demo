# Agentic RAG System

A complete agentic RAG (Retrieval-Augmented Generation) system with OpenWebUI frontend, featuring document processing, vector storage, agentic orchestration, observability, and evaluation capabilities.

## Overview

This system implements a production-ready RAG pipeline with the following key features:

- **Document Processing**: Multi-format document ingestion using Docling (PDF, DOCX, TXT, Markdown)
- **Vector Storage**: PGVector/PostgreSQL for efficient similarity search
- **Agentic Orchestration**: Crew.AI-powered multi-agent system for query understanding, retrieval, and answer generation
- **Conversation Memory**: PostgreSQL-based conversation history management
- **Citation Handling**: Automatic extraction and formatting of source citations
- **Observability**: Arize Phoenix integration for LLM tracing and debugging
- **Evaluation**: RAGAs framework for comprehensive RAG system evaluation
- **Frontend**: OpenWebUI Docker deployment with OpenAI-compatible API

## Architecture

### System Components

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  OpenWebUI  │───▶│  FastAPI    │───▶│  Crew.AI    │
│  Frontend   │    │  Backend    │    │  Orchestrator│
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
            ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
            │ Query Agent  │         │Retrieval    │         │ Answer Agent │
            │              │         │Agent        │         │              │
            └──────────────┘         └──────────────┘         └──────────────┘
                                                                    │
                    ┌───────────────────────────────────────────────┘
                    │
                    ▼
            ┌──────────────┐
            │  PGVector    │
            │  + Re-ranker │
            └──────────────┘
```

### Data Flow

1. **Indexing**: Documents → Docling → Semantic Chunking → Embeddings → PGVector
2. **Query**: User Query → Query Agent → Retrieval Agent → Re-ranking → Answer Agent → Response with Citations
3. **Memory**: Conversations stored in PostgreSQL, retrieved for context
4. **Observability**: All LLM calls traced through Phoenix

## Technology Stack

- **Document Processing**: Docling
- **RAG Framework**: LlamaIndex
- **Vector Database**: PGVector (PostgreSQL extension)
- **Agentic Framework**: Crew.AI
- **LLM Hosting**: Ollama
- **Observability**: Arize Phoenix
- **Evaluation**: RAGAs
- **Frontend**: OpenWebUI
- **Backend**: FastAPI
- **Database**: PostgreSQL
- **Deployment**: Docker Compose

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- At least 16GB RAM (for running Ollama models)
- 50GB+ disk space (for models and data)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agentic-rag-system
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration (defaults should work for local development).

### 3. Start Services

Start all services using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- PostgreSQL with pgvector
- Ollama (LLM hosting)
- Phoenix (Observability)
- OpenWebUI (Frontend)
- Backend API

### 4. Setup Ollama Models

Pull required models:

```bash
docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull nomic-embed-text
```

Or use the setup script:

```bash
docker-compose exec ollama bash /app/scripts/setup_ollama.sh
```

### 5. Index Documents

Index documents from a directory:

```bash
docker-compose run --rm indexer python -m src.indexer.main /path/to/documents
```

Or index a single file:

```bash
docker-compose run --rm indexer python -m src.indexer.main /path/to/document.pdf
```

### 6. Access the System

- **OpenWebUI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Phoenix**: http://localhost:6006

## Project Structure

```
agentic-rag-system/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── .gitignore
├── src/
│   ├── indexer/          # Document indexer
│   ├── backend/           # FastAPI backend
│   │   ├── api/          # API routes and schemas
│   │   ├── agents/       # Crew.AI agents
│   │   ├── rag/          # RAG components
│   │   ├── memory/       # Conversation memory
│   │   ├── citations/    # Citation handling
│   │   └── observability/# Phoenix integration
│   ├── evaluator/         # RAGAs evaluator
│   └── config/           # Configuration
├── tests/                 # Unit tests
├── scripts/               # Utility scripts
└── docker/                # Dockerfiles
```

## Usage

### Document Indexing

Index documents using the console application:

```bash
# Index a directory
python -m src.indexer.main /path/to/documents

# Index a single file
python -m src.indexer.main /path/to/document.pdf

# Non-recursive directory indexing
python -m src.indexer.main /path/to/documents --no-recursive
```

### API Usage

#### Chat Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "conversation_id": null,
    "stream": false
  }'
```

#### Create Conversation

```bash
curl -X POST http://localhost:8000/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123"}'
```

#### Get Conversation History

```bash
curl http://localhost:8000/api/v1/conversations/{conversation_id}
```

#### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Evaluation

Run RAGAs evaluation:

```bash
# Create evaluation questions file (questions.json)
[
  "What is machine learning?",
  "Explain neural networks",
  "What is the difference between AI and ML?"
]

# Run evaluation
python -m src.evaluator.main questions.json --output evaluation_report.txt
```

With ground truths:

```bash
# Create ground truths file (ground_truths.json)
[
  "Machine learning is a subset of AI...",
  "Neural networks are computing systems...",
  "AI is broader than ML..."
]

# Run evaluation with ground truths
python -m src.evaluator.main questions.json \
  --ground-truths ground_truths.json \
  --output evaluation_report.txt
```

## Configuration

### Environment Variables

Key configuration options in `.env`:

- `POSTGRES_*`: Database connection settings
- `OLLAMA_BASE_URL`: Ollama service URL
- `OLLAMA_MODEL`: LLM model name (default: llama3.1:8b)
- `EMBEDDING_MODEL_NAME`: Embedding model (default: BAAI/bge-small-en-v1.5)
- `PHOENIX_HOST`, `PHOENIX_PORT`: Phoenix observability settings
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Document chunking parameters
- `TOP_K_RETRIEVAL`, `TOP_K_RERANK`: Retrieval and re-ranking settings

### Design Choices

- **Chunking Strategy**: Semantic chunking with 512 tokens, 50 token overlap
- **Embedding Model**: BAAI/bge-small-en-v1.5 (384 dimensions)
- **LLM Model**: llama3.1:8b via Ollama
- **Retrieval**: Top 10 candidates, re-ranked to top 3
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2

## API Documentation

### Endpoints

- `POST /api/v1/chat` - Chat endpoint
- `GET /api/v1/health` - Health check
- `POST /api/v1/conversations` - Create conversation
- `GET /api/v1/conversations/{id}` - Get conversation history
- `POST /api/v1/documents/ingest` - Manual document ingestion
- `POST /v1/chat/completions` - OpenAI-compatible endpoint for OpenWebUI

Full API documentation available at http://localhost:8000/docs

## Testing

Run tests:

```bash
# All tests
pytest

# Specific test file
pytest tests/test_backend.py

# With coverage
pytest --cov=src tests/
```

## Development

### Local Development Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start services (database, Ollama, Phoenix):

```bash
docker-compose up -d postgres ollama phoenix
```

3. Run backend locally:

```bash
uvicorn src.backend.main:app --reload
```

### Adding New Documents

Place documents in `data/documents/` directory, then index:

```bash
python -m src.indexer.main data/documents
```

## Troubleshooting

### Ollama Models Not Loading

Ensure models are pulled:

```bash
docker-compose exec ollama ollama list
docker-compose exec ollama ollama pull llama3.1:8b
```

### Database Connection Issues

Check PostgreSQL is running:

```bash
docker-compose ps postgres
docker-compose logs postgres
```

### Phoenix Not Accessible

Check Phoenix logs:

```bash
docker-compose logs phoenix
```

## Performance Considerations

- **Embedding Generation**: First run may be slow as models download
- **LLM Inference**: Response time depends on hardware (GPU recommended)
- **Vector Search**: Optimized with PGVector indexes
- **Re-ranking**: Adds ~100-200ms per query

## License

This project is 100% open source.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions, please open an issue on the repository.

## Acknowledgments

Built with:
- [LlamaIndex](https://www.llamaindex.ai/)
- [Crew.AI](https://www.crewai.com/)
- [Arize Phoenix](https://arize.com/phoenix/)
- [RAGAs](https://docs.ragas.io/)
- [OpenWebUI](https://github.com/open-webui/open-webui)
- [Ollama](https://ollama.ai/)

