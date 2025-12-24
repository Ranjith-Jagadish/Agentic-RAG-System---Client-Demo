"""Re-ranker using cross-encoder model"""

from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import CrossEncoder
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder re-ranker for improving retrieval relevance"""
    
    def __init__(self):
        """Initialize the re-ranker"""
        logger.info("Initializing cross-encoder re-ranker")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Re-ranker initialized")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on query relevance
        
        Args:
            query: Query string
            documents: List of document dictionaries with 'text' key
            top_k: Number of top results to return (defaults to settings)
            
        Returns:
            Re-ranked list of documents
        """
        if not documents:
            return []
        
        try:
            if top_k is None:
                top_k = settings.top_k_rerank
            
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.get("text", "")] for doc in documents]
            
            # Get scores
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            scored_docs = [
                {**doc, "rerank_score": float(score)}
                for doc, score in zip(documents, scores)
            ]
            
            # Sort by rerank score (descending)
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top_k
            reranked = scored_docs[:top_k]
            
            logger.info(f"Re-ranked {len(documents)} documents to top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.error(f"Error re-ranking documents: {str(e)}")
            # Return original documents if re-ranking fails
            return documents[:top_k] if top_k else documents

