"""Vector store implementation using LlamaIndex and PGVector"""

from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import create_engine
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """PGVector store for document embeddings"""
    
    def __init__(self):
        """Initialize the vector store"""
        self.embedding_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model_name
        )
        
        # Create database connection string
        connection_string = settings.database_url
        
        # Create engine
        self.engine = create_engine(connection_string)
        
        # Initialize vector store
        self.vector_store = PGVectorStore.from_params(
            database=settings.postgres_db,
            host=settings.postgres_host,
            password=settings.postgres_password,
            port=settings.postgres_port,
            user=settings.postgres_user,
            table_name="document_vectors",
            embed_dim=settings.embedding_dimension,
        )
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Initialize index
        self.index = None
    
    def create_index(self, documents: Optional[List[LlamaDocument]] = None) -> VectorStoreIndex:
        """
        Create or get vector store index
        
        Args:
            documents: Optional list of documents to index
            
        Returns:
            VectorStoreIndex instance
        """
        if documents:
            logger.info(f"Creating index with {len(documents)} documents")
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                embed_model=self.embedding_model,
                show_progress=True
            )
        else:
            logger.info("Loading existing index")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embedding_model
            )
        
        return self.index
    
    def add_documents(self, documents: List[LlamaDocument]) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents to add
        """
        if not self.index:
            self.index = self.create_index(documents)
        else:
            logger.info(f"Adding {len(documents)} documents to existing index")
            for doc in documents:
                self.index.insert(doc)
    
    def get_index(self) -> VectorStoreIndex:
        """Get the vector store index"""
        if not self.index:
            self.index = self.create_index()
        return self.index
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store
        
        Args:
            document_ids: List of document IDs to delete
        """
        if not self.index:
            logger.warning("Index not initialized, cannot delete documents")
            return
        
        logger.info(f"Deleting {len(document_ids)} documents")
        # Note: LlamaIndex PGVectorStore deletion would need to be implemented
        # based on the specific version and API
    
    def clear_index(self) -> None:
        """Clear all documents from the vector store"""
        logger.warning("Clearing entire vector store")
        # This would require dropping and recreating the table
        # Implementation depends on specific requirements

