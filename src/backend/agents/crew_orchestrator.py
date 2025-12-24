"""Crew.AI orchestrator for agentic RAG"""

from typing import Dict, Any, List, Optional
from crewai import Crew, Task
from src.backend.agents.query_agent import create_query_agent
from src.backend.agents.retrieval_agent import create_retrieval_agent
from src.backend.agents.answer_agent import create_answer_agent
from src.backend.rag.retriever import RAGRetriever
from src.backend.rag.reranker import Reranker
from src.backend.rag.llm_service import LLMService
from src.backend.citations.citation_handler import CitationHandler
from src.backend.observability.phoenix_integration import PhoenixIntegration
import logging

logger = logging.getLogger(__name__)


class CrewOrchestrator:
    """Orchestrates Crew.AI agents for RAG pipeline"""
    
    def __init__(self):
        """Initialize the orchestrator"""
        # Initialize components
        self.retriever = RAGRetriever()
        self.reranker = Reranker()
        self.llm_service = LLMService()
        self.citation_handler = CitationHandler()
        self.phoenix = PhoenixIntegration()
        
        # Create agents
        self.query_agent = create_query_agent(self.llm_service, self.phoenix)
        self.retrieval_agent = create_retrieval_agent(
            self.retriever,
            self.reranker,
            self.phoenix
        )
        self.answer_agent = create_answer_agent(
            self.llm_service,
            self.citation_handler,
            self.phoenix
        )
        
        logger.info("Crew.AI orchestrator initialized")
    
    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the agentic RAG pipeline
        
        Args:
            query: User query
            conversation_history: Optional conversation history for context
            
        Returns:
            Dictionary with response, citations, and metadata
        """
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Build context from conversation history
            context = self._build_context(conversation_history)
            
            # Task 1: Query Understanding
            query_task = Task(
                description=f"""Analyze the following user query and extract key information:
                
Query: {query}

{context}

Provide a structured analysis of the query intent, entities, and required information type.""",
                agent=self.query_agent,
                expected_output="Structured analysis of query intent and key concepts"
            )
            
            # Task 2: Document Retrieval
            retrieval_task = Task(
                description=f"""Based on the query analysis, retrieve the most relevant document chunks.
                
Query: {query}

Use semantic search to find relevant documents and re-rank them for relevance.""",
                agent=self.retrieval_agent,
                expected_output="List of relevant document chunks with scores",
                context=[query_task]
            )
            
            # Task 3: Answer Generation
            answer_task = Task(
                description=f"""Generate a comprehensive answer based on the retrieved documents.
                
Query: {query}

Retrieved Documents: [Use the results from the retrieval task]

Generate an accurate answer with proper citations to source documents.""",
                agent=self.answer_agent,
                expected_output="Comprehensive answer with citations",
                context=[query_task, retrieval_task]
            )
            
            # Create crew
            crew = Crew(
                agents=[self.query_agent, self.retrieval_agent, self.answer_agent],
                tasks=[query_task, retrieval_task, answer_task],
                verbose=True
            )
            
            # Execute crew (run in thread pool for async compatibility)
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, crew.kickoff)
            
            # Extract answer from result
            answer = str(result) if result else "I apologize, but I couldn't generate an answer."
            
            # Retrieve documents for citations
            nodes = self.retriever.retrieve(query)
            reranked_docs = self.reranker.rerank(
                query,
                self.retriever.format_nodes(nodes)
            )
            
            # Extract citations
            citations = self.citation_handler.extract_citations(
                [node for node in nodes if node.node.node_id in [doc.get("node_id") for doc in reranked_docs]]
            )
            
            # Trace agent execution
            self.phoenix.trace_agent_execution(
                agent_name="crew_orchestrator",
                task="process_query",
                result=answer,
                metadata={"query": query, "citations_count": len(citations)}
            )
            
            return {
                "response": answer,
                "citations": [cit.dict() for cit in citations],
                "metadata": {
                    "query": query,
                    "retrieved_count": len(reranked_docs),
                    "citations_count": len(citations)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise
    
    def _build_context(self, conversation_history: Optional[List[Dict[str, Any]]]) -> str:
        """Build context string from conversation history"""
        if not conversation_history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_parts.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(context_parts)

