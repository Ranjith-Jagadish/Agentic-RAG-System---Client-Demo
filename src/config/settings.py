"""Configuration settings for the agentic RAG system"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Database Configuration
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "raguser"
    postgres_password: str = "ragpassword"
    postgres_db: str = "ragdb"
    
    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # Ollama Configuration
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_embedding_model: str = "nomic-embed-text"
    
    # Embedding Model
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    
    # Phoenix Configuration
    phoenix_host: str = "localhost"
    phoenix_port: int = 6006
    phoenix_collection_name: str = "rag_system"
    
    @property
    def phoenix_url(self) -> str:
        """Construct Phoenix URL"""
        return f"http://{self.phoenix_host}:{self.phoenix_port}"
    
    # RAG Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # OpenWebUI Configuration
    openwebui_url: str = "http://openwebui:3000"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

