"""Semantic chunking strategy for documents"""

from typing import List, Dict, Any
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """Semantic chunking strategy using LlamaIndex"""
    
    def __init__(self):
        """Initialize the chunking strategy"""
        # Initialize embedding model for semantic chunking
        self.embedding_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model_name
        )
        
        # Initialize semantic splitter
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embedding_model,
        )
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[LlamaDocument]:
        """
        Chunk a document using semantic chunking
        
        Args:
            text: Document text content
            metadata: Document metadata
            
        Returns:
            List of chunked LlamaIndex Document objects
        """
        logger.info(f"Chunking document: {metadata.get('file_name', 'unknown')}")
        
        # Create LlamaIndex document
        llama_doc = LlamaDocument(
            text=text,
            metadata=metadata
        )
        
        # Split into nodes
        nodes = self.splitter.get_nodes_from_documents([llama_doc])
        
        # Convert nodes back to documents with metadata
        chunks = []
        for i, node in enumerate(nodes):
            chunk_metadata = {
                **metadata,
                "chunk_id": f"{metadata.get('file_name', 'doc')}_chunk_{i}",
                "chunk_index": i,
                "total_chunks": len(nodes)
            }
            
            chunk_doc = LlamaDocument(
                text=node.text,
                metadata=chunk_metadata
            )
            chunks.append(chunk_doc)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[LlamaDocument]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            List of chunked LlamaIndex Document objects
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.chunk_document(
                    text=doc["text"],
                    metadata=doc["metadata"]
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {doc.get('metadata', {}).get('file_name', 'unknown')}: {str(e)}")
                continue
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

