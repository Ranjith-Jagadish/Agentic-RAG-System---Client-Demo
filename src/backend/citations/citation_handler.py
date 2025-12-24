"""Citation handler for extracted metadata"""

from typing import List, Dict, Any
from llama_index.core.schema import NodeWithScore
from src.backend.api.schemas import Citation
import logging

logger = logging.getLogger(__name__)


class CitationHandler:
    """Handler for extracting and formatting citations"""
    
    @staticmethod
    def extract_citations(nodes: List[NodeWithScore]) -> List[Citation]:
        """
        Extract citations from retrieved nodes
        
        Args:
            nodes: List of NodeWithScore objects
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for node in nodes:
            metadata = node.node.metadata
            citation = Citation(
                chunk_id=node.node.node_id,
                document_name=metadata.get("file_name", "Unknown"),
                page_number=metadata.get("page_number"),
                score=float(node.score) if node.score else 0.0,
                text=node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text
            )
            citations.append(citation)
        
        return citations
    
    @staticmethod
    def format_citations_for_prompt(citations: List[Citation]) -> str:
        """
        Format citations for inclusion in LLM prompt
        
        Args:
            citations: List of Citation objects
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return ""
        
        formatted = "\n\nSources:\n"
        for i, citation in enumerate(citations, 1):
            formatted += f"[{i}] {citation.document_name}"
            if citation.page_number:
                formatted += f" (Page {citation.page_number})"
            formatted += f"\n"
        
        return formatted
    
    @staticmethod
    def format_context_with_citations(
        nodes: List[NodeWithScore],
        include_scores: bool = False
    ) -> str:
        """
        Format retrieved context with citations
        
        Args:
            nodes: List of NodeWithScore objects
            include_scores: Whether to include similarity scores
            
        Returns:
            Formatted context string
        """
        if not nodes:
            return ""
        
        formatted = "Context from documents:\n\n"
        for i, node in enumerate(nodes, 1):
            metadata = node.node.metadata
            doc_name = metadata.get("file_name", "Unknown")
            page_num = metadata.get("page_number")
            
            formatted += f"[Source {i}] {doc_name}"
            if page_num:
                formatted += f" (Page {page_num})"
            if include_scores and node.score:
                formatted += f" [Score: {node.score:.3f}]"
            formatted += ":\n"
            formatted += f"{node.node.text}\n\n"
        
        return formatted

