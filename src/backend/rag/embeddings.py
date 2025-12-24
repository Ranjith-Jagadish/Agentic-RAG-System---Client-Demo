"""Embedding service using HuggingFace models"""

from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings"""
    
    def __init__(self):
        """Initialize the embedding model"""
        logger.info(f"Initializing embedding model: {settings.embedding_model_name}")
        self.model = HuggingFaceEmbedding(
            model_name=settings.embedding_model_name
        )
        logger.info("Embedding model initialized")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.get_text_embedding_batch(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.get_text_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return settings.embedding_dimension

