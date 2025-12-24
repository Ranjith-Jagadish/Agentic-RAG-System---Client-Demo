"""Pydantic schemas for API requests and responses"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Citation(BaseModel):
    """Citation metadata for retrieved chunks"""
    chunk_id: str
    document_name: str
    page_number: Optional[int] = None
    score: float
    text: Optional[str] = None


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    stream: bool = Field(False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    response: str = Field(..., description="AI generated response")
    conversation_id: str = Field(..., description="Conversation ID")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationCreate(BaseModel):
    """Request schema for creating a new conversation"""
    user_id: Optional[str] = Field(None, description="Optional user identifier")


class ConversationResponse(BaseModel):
    """Response schema for conversation"""
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class Message(BaseModel):
    """Message schema"""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history"""
    conversation_id: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, str] = Field(default_factory=dict)


class DocumentIngestRequest(BaseModel):
    """Request schema for document ingestion"""
    file_path: str = Field(..., description="Path to document file")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentIngestResponse(BaseModel):
    """Response schema for document ingestion"""
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_created: int = 0

