"""LLM service using Ollama"""

from typing import List, Optional, Dict, Any
from llama_index.llms.ollama import Ollama
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM interactions via Ollama"""
    
    def __init__(self):
        """Initialize the LLM service"""
        logger.info(f"Initializing Ollama LLM: {settings.ollama_model}")
        self.llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.7,
            request_timeout=120.0
        )
        logger.info("Ollama LLM initialized")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            response = self.llm.complete(prompt, **kwargs)
            return str(response)
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_stream(self, prompt: str, **kwargs):
        """
        Generate text stream from a prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        try:
            response_stream = self.llm.stream_complete(prompt, **kwargs)
            for chunk in response_stream:
                yield str(chunk)
        except Exception as e:
            logger.error(f"Error streaming text: {str(e)}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat completion
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        try:
            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages)
            return self.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"

