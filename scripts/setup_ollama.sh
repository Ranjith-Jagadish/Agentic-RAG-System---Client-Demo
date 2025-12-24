#!/bin/bash
set -e

echo "Setting up Ollama models..."

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# Pull required models
echo "Pulling llama3.1:8b model..."
curl -X POST "$OLLAMA_URL/api/pull" -d '{"name": "llama3.1:8b"}'

echo "Pulling nomic-embed-text model..."
curl -X POST "$OLLAMA_URL/api/pull" -d '{"name": "nomic-embed-text"}'

echo "Ollama models setup completed!"

