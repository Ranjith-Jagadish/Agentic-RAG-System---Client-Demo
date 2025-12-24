"""Query understanding agent using Crew.AI"""

from crewai import Agent
from crewai.tools import Tool
from typing import Dict, Any
from src.backend.rag.llm_service import LLMService
from src.backend.observability.phoenix_integration import PhoenixIntegration
import logging

logger = logging.getLogger(__name__)


class QueryUnderstandingTool(Tool):
    """Tool for query understanding"""
    
    def __init__(self, llm_service: LLMService, phoenix: PhoenixIntegration):
        super().__init__(
            name="query_understanding",
            description="Analyze user query to extract intent, entities, and key concepts"
        )
        self.llm_service = llm_service
        self.phoenix = phoenix
    
    def _run(self, query: str) -> str:
        """Execute query understanding"""
        prompt = self.phoenix.get_prompt("query_understanding") or """Analyze the following user query and extract:
1. Main intent
2. Key entities and concepts
3. Required information type

Query: {query}

Provide a structured analysis."""
        
        formatted_prompt = prompt.format(query=query)
        
        try:
            analysis = self.llm_service.generate(formatted_prompt)
            self.phoenix.trace_llm_call(
                prompt=formatted_prompt,
                response=analysis,
                metadata={"tool": "query_understanding"},
                model_name="query_analyzer"
            )
            return analysis
        except Exception as e:
            logger.error(f"Error in query understanding: {str(e)}")
            return f"Query analysis error: {str(e)}"


def create_query_agent(llm_service: LLMService, phoenix: PhoenixIntegration) -> Agent:
    """Create the query understanding agent"""
    
    query_tool = QueryUnderstandingTool(llm_service, phoenix)
    
    agent = Agent(
        role="Query Understanding Specialist",
        goal="Analyze user queries to extract intent, entities, and key concepts for effective information retrieval",
        backstory="""You are an expert at understanding user queries. You break down complex questions
        into their core components, identify the user's intent, extract key entities and concepts,
        and determine what type of information is needed.""",
        tools=[query_tool],
        verbose=True,
        allow_delegation=False
    )
    
    return agent

