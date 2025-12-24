"""Document retrieval agent using Crew.AI"""

from crewai import Agent
from crewai.tools import Tool
from typing import List, Dict, Any
from src.backend.rag.retriever import RAGRetriever
from src.backend.rag.reranker import Reranker
from src.backend.observability.phoenix_integration import PhoenixIntegration
import logging

logger = logging.getLogger(__name__)


class DocumentRetrievalTool(Tool):
    """Tool for document retrieval"""
    
    def __init__(
        self,
        retriever: RAGRetriever,
        reranker: Reranker,
        phoenix: PhoenixIntegration
    ):
        super().__init__(
            name="document_retrieval",
            description="Retrieve relevant document chunks from the vector database"
        )
        self.retriever = retriever
        self.reranker = reranker
        self.phoenix = phoenix
    
    def _run(self, query: str) -> str:
        """Execute document retrieval"""
        try:
            # Retrieve documents
            nodes = self.retriever.retrieve(query)
            
            # Format nodes for re-ranking
            formatted_docs = self.retriever.format_nodes(nodes)
            
            # Re-rank documents
            reranked_docs = self.reranker.rerank(query, formatted_docs)
            
            # Format results
            result = f"Retrieved {len(reranked_docs)} relevant document chunks:\n\n"
            for i, doc in enumerate(reranked_docs, 1):
                result += f"[{i}] {doc.get('metadata', {}).get('file_name', 'Unknown')}\n"
                result += f"Score: {doc.get('rerank_score', 0):.3f}\n"
                result += f"Content: {doc.get('text', '')[:200]}...\n\n"
            
            # Trace retrieval
            self.phoenix.trace_retrieval(
                query=query,
                retrieved_docs=reranked_docs,
                metadata={"count": len(reranked_docs)}
            )
            
            return result
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            return f"Retrieval error: {str(e)}"


def create_retrieval_agent(
    retriever: RAGRetriever,
    reranker: Reranker,
    phoenix: PhoenixIntegration
) -> Agent:
    """Create the document retrieval agent"""
    
    retrieval_tool = DocumentRetrievalTool(retriever, reranker, phoenix)
    
    agent = Agent(
        role="Document Retrieval Specialist",
        goal="Retrieve the most relevant document chunks from the knowledge base for user queries",
        backstory="""You are an expert at finding relevant information in large document collections.
        You use semantic search and re-ranking to identify the most pertinent document chunks that
        can answer user questions accurately.""",
        tools=[retrieval_tool],
        verbose=True,
        allow_delegation=False
    )
    
    return agent

