"""Tests for document indexer"""

import pytest
from pathlib import Path
from src.indexer.document_processor import DocumentProcessor
from src.indexer.chunking_strategy import ChunkingStrategy


def test_document_processor_initialization():
    """Test document processor initialization"""
    processor = DocumentProcessor()
    assert processor is not None
    assert processor.converter is not None


def test_chunking_strategy_initialization():
    """Test chunking strategy initialization"""
    chunker = ChunkingStrategy()
    assert chunker is not None
    assert chunker.embedding_model is not None
    assert chunker.splitter is not None


def test_chunk_document():
    """Test document chunking"""
    chunker = ChunkingStrategy()
    
    test_text = "This is a test document. " * 100
    metadata = {"file_name": "test.txt", "file_type": "text"}
    
    chunks = chunker.chunk_document(test_text, metadata)
    
    assert len(chunks) > 0
    assert all(chunk.metadata.get("file_name") == "test.txt" for chunk in chunks)


@pytest.mark.skip(reason="Requires database connection")
def test_vector_store_initialization():
    """Test vector store initialization"""
    from src.indexer.vector_store import VectorStore
    
    vector_store = VectorStore()
    assert vector_store is not None
    assert vector_store.embedding_model is not None

