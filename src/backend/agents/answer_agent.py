"""Answer generation agent using Crew.AI"""

from crewai import Agent
from crewai.tools import Tool
from typing import Dict, Any
from src.backend.rag.llm_service import LLMService
from src.backend.citations.citation_handler import CitationHandler
from src.backend.observability.phoenix_integration import PhoenixIntegration
import logging

logger = logging.getLogger(__name__)


class AnswerGenerationTool(Tool):
    """Tool for answer generation"""
    
    def __init__(
        self,
        llm_service: LLMService,
        citation_handler: CitationHandler,
        phoenix: PhoenixIntegration
    ):
        super().__init__(
            name="answer_generation",
            description="Generate accurate answers based on retrieved context with proper citations"
        )
        self.llm_service = llm_service
        self.citation_handler = citation_handler
        self.phoenix = phoenix
    
    def _run(self, query: str, context: str, citations: str = "") -> str:
        """Execute answer generation"""
        prompt_template = self.phoenix.get_prompt("answer_generation") or """Based on the following context, answer the user's question.
Provide a clear, accurate answer with citations to sources.

Context:
{context}

{citations}

Question: {query}

Answer:"""
        
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            citations=citations
        )
        
        try:
            answer = self.llm_service.generate(formatted_prompt)
            self.phoenix.trace_llm_call(
                prompt=formatted_prompt,
                response=answer,
                metadata={"tool": "answer_generation"},
                model_name="answer_generator"
            )
            return answer
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            return f"Answer generation error: {str(e)}"


def create_answer_agent(
    llm_service: LLMService,
    citation_handler: CitationHandler,
    phoenix: PhoenixIntegration
) -> Agent:
    """Create the answer generation agent"""
    
    answer_tool = AnswerGenerationTool(llm_service, citation_handler, phoenix)
    
    agent = Agent(
        role="Answer Generation Specialist",
        goal="Generate accurate, well-cited answers based on retrieved document context",
        backstory="""You are an expert at synthesizing information from multiple sources to create
        comprehensive answers. You always cite your sources and ensure accuracy. You provide clear,
        well-structured responses that directly address user questions.""",
        tools=[answer_tool],
        verbose=True,
        allow_delegation=False
    )
    
    return agent

