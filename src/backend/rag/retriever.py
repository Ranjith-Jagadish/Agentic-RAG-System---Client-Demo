"""Retriever using LlamaIndex and PGVector"""

from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.indexer.vector_store import VectorStore
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Retriever for document chunks from PGVector"""
    
    def __init__(self):
        """Initialize the retriever"""
        self.vector_store = VectorStore()
        self.index = self.vector_store.get_index()
        self.retriever = self.index.as_retriever(
            similarity_top_k=settings.top_k_retrieval
        )
        logger.info("RAG Retriever initialized")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve relevant document chunks
        
        Args:
            query: Query string
            top_k: Number of results to retrieve (defaults to settings)
            
        Returns:
            List of NodeWithScore objects
        """
        try:
            if top_k:
                # Create temporary retriever with custom top_k
                temp_retriever = self.index.as_retriever(similarity_top_k=top_k)
                nodes = temp_retriever.retrieve(query)
            else:
                nodes = self.retriever.retrieve(query)
            
            logger.info(f"Retrieved {len(nodes)} chunks for query: {query[:50]}...")
            return nodes
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def format_nodes(self, nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """
        Format retrieved nodes for use in prompts
        
        Args:
            nodes: List of NodeWithScore objects
            
        Returns:
            List of formatted dictionaries
        """
        formatted = []
        for i, node in enumerate(nodes):
            formatted.append({
                "text": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata,
                "node_id": node.node.node_id
            })
        return formatted

