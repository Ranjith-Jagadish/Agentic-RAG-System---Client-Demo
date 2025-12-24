"""Arize Phoenix integration for observability and prompt management"""

from typing import Dict, Any, Optional, List
from phoenix.otel import register
from phoenix.session import Session
from phoenix.trace import SpanKind, SpanStatusCode
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class PhoenixIntegration:
    """Arize Phoenix integration for tracing and observability"""
    
    def __init__(self):
        """Initialize Phoenix integration"""
        self.session = None
        self.instrumentor = None
        self._initialize_phoenix()
    
    def _initialize_phoenix(self):
        """Initialize Phoenix session and instrumentation"""
        try:
            # Create Phoenix session
            self.session = Session(
                project_name=settings.phoenix_collection_name,
                host=settings.phoenix_host,
                port=settings.phoenix_port
            )
            
            # Register OpenTelemetry
            register(self.session)
            
            # Instrument LlamaIndex
            self.instrumentor = LlamaIndexInstrumentor()
            self.instrumentor.instrument()
            
            logger.info("Phoenix integration initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Phoenix: {str(e)}")
            self.session = None
    
    def trace_llm_call(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> None:
        """
        Trace an LLM call
        
        Args:
            prompt: Input prompt
            response: LLM response
            metadata: Optional metadata
            model_name: Model name
        """
        if not self.session:
            return
        
        try:
            from phoenix.trace import trace
            
            with trace(
                name="llm_call",
                kind=SpanKind.LLM,
                attributes={
                    "llm.request.type": "completion",
                    "llm.request.model": model_name or settings.ollama_model,
                    "llm.prompt": prompt,
                    "llm.response": response,
                    **(metadata or {})
                }
            ):
                pass  # Trace is automatically recorded
        except Exception as e:
            logger.warning(f"Failed to trace LLM call: {str(e)}")
    
    def trace_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Trace a retrieval operation
        
        Args:
            query: Query string
            retrieved_docs: List of retrieved documents
            metadata: Optional metadata
        """
        if not self.session:
            return
        
        try:
            from phoenix.trace import trace
            
            with trace(
                name="retrieval",
                kind=SpanKind.RETRIEVER,
                attributes={
                    "retrieval.query": query,
                    "retrieval.documents.count": len(retrieved_docs),
                    "retrieval.documents": str(retrieved_docs),
                    **(metadata or {})
                }
            ):
                pass
        except Exception as e:
            logger.warning(f"Failed to trace retrieval: {str(e)}")
    
    def trace_agent_execution(
        self,
        agent_name: str,
        task: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Trace an agent execution
        
        Args:
            agent_name: Name of the agent
            task: Task description
            result: Agent result
            metadata: Optional metadata
        """
        if not self.session:
            return
        
        try:
            from phoenix.trace import trace
            
            with trace(
                name=f"agent_{agent_name}",
                kind=SpanKind.AGENT,
                attributes={
                    "agent.name": agent_name,
                    "agent.task": task,
                    "agent.result": result,
                    **(metadata or {})
                }
            ):
                pass
        except Exception as e:
            logger.warning(f"Failed to trace agent execution: {str(e)}")
    
    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Retrieve a prompt from Phoenix
        
        Args:
            prompt_name: Name of the prompt
            
        Returns:
            Prompt template or None
        """
        # Note: This is a placeholder. Actual prompt retrieval would depend on
        # Phoenix's prompt management API, which may vary by version.
        # In practice, prompts might be stored in Phoenix or retrieved via API.
        logger.info(f"Retrieving prompt: {prompt_name}")
        # Return default prompts for now
        return self._get_default_prompt(prompt_name)
    
    def _get_default_prompt(self, prompt_name: str) -> Optional[str]:
        """Get default prompt templates"""
        prompts = {
            "query_understanding": """Analyze the following user query and extract:
1. Main intent
2. Key entities and concepts
3. Required information type

Query: {query}

Provide a structured analysis.""",
            
            "answer_generation": """Based on the following context, answer the user's question.
Provide a clear, accurate answer with citations to sources.

Context:
{context}

Question: {query}

Answer:""",
            
            "retrieval": """Retrieve relevant documents for the following query:
Query: {query}

Return the most relevant document chunks."""
        }
        return prompts.get(prompt_name)
    
    def flush(self):
        """Flush traces to Phoenix"""
        if self.session:
            try:
                self.session.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Phoenix traces: {str(e)}")

