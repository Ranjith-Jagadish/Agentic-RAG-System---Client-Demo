"""Tests for RAG components"""

import pytest


def test_embedding_service_initialization():
    """Test embedding service initialization"""
    from src.backend.rag.embeddings import EmbeddingService
    
    service = EmbeddingService()
    assert service is not None
    assert service.model is not None
    assert service.dimension > 0


def test_embedding_generation():
    """Test embedding generation"""
    from src.backend.rag.embeddings import EmbeddingService
    
    service = EmbeddingService()
    text = "This is a test sentence."
    
    embedding = service.get_embedding(text)
    
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) == service.dimension


def test_llm_service_initialization():
    """Test LLM service initialization"""
    from src.backend.rag.llm_service import LLMService
    
    service = LLMService()
    assert service is not None
    assert service.llm is not None


@pytest.mark.skip(reason="Requires Ollama connection")
def test_llm_generation():
    """Test LLM text generation"""
    from src.backend.rag.llm_service import LLMService
    
    service = LLMService()
    prompt = "Say hello in one sentence."
    
    response = service.generate(prompt)
    
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


def test_reranker_initialization():
    """Test reranker initialization"""
    from src.backend.rag.reranker import Reranker
    
    reranker = Reranker()
    assert reranker is not None
    assert reranker.model is not None


def test_reranking():
    """Test document reranking"""
    from src.backend.rag.reranker import Reranker
    
    reranker = Reranker()
    query = "What is machine learning?"
    
    documents = [
        {"text": "Machine learning is a subset of artificial intelligence.", "metadata": {}},
        {"text": "The weather today is sunny.", "metadata": {}},
        {"text": "Deep learning uses neural networks.", "metadata": {}}
    ]
    
    reranked = reranker.rerank(query, documents, top_k=2)
    
    assert len(reranked) == 2
    assert all("rerank_score" in doc for doc in reranked)
    # First document should be more relevant
    assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]


def test_citation_handler():
    """Test citation handler"""
    from src.backend.citations.citation_handler import CitationHandler
    from llama_index.core.schema import NodeWithScore, TextNode
    
    handler = CitationHandler()
    
    # Create mock nodes
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="Test content",
                metadata={"file_name": "test.pdf", "page_number": 1}
            ),
            score=0.95
        )
    ]
    
    citations = handler.extract_citations(nodes)
    
    assert len(citations) == 1
    assert citations[0].document_name == "test.pdf"
    assert citations[0].page_number == 1
    assert citations[0].score == 0.95

