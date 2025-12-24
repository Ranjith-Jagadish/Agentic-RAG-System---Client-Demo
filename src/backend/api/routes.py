"""API routes for the backend"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from src.backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationCreate,
    ConversationResponse,
    ConversationHistoryResponse,
    HealthResponse,
    DocumentIngestRequest,
    DocumentIngestResponse,
)
from src.backend.memory.conversation_memory import ConversationMemory
from src.backend.agents.crew_orchestrator import CrewOrchestrator
from src.config.settings import settings
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_prefix)

# OpenAI-compatible schemas for OpenWebUI
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIRequest(BaseModel):
    model: str = "llama3.1:8b"
    messages: List[OpenAIMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"

class OpenAIResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check database connection
    try:
        memory = ConversationMemory()
        await memory.check_connection()
        services["database"] = "healthy"
    except Exception as e:
        services["database"] = f"unhealthy: {str(e)}"
    
    # Check Ollama
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
            if response.status_code == 200:
                services["ollama"] = "healthy"
            else:
                services["ollama"] = "unhealthy"
    except Exception as e:
        services["ollama"] = f"unhealthy: {str(e)}"
    
    # Check Phoenix
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.phoenix_url}/health", timeout=5.0)
            if response.status_code == 200:
                services["phoenix"] = "healthy"
            else:
                services["phoenix"] = "unhealthy"
    except Exception as e:
        services["phoenix"] = f"unhealthy: {str(e)}"
    
    all_healthy = all("healthy" in status for status in services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.now(),
        services=services
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for OpenWebUI"""
    try:
        # Initialize orchestrator
        orchestrator = CrewOrchestrator()
        
        # Get or create conversation
        memory = ConversationMemory()
        if not request.conversation_id:
            conversation = await memory.create_conversation()
            conversation_id = conversation["conversation_id"]
        else:
            conversation_id = request.conversation_id
        
        # Save user message
        await memory.add_message(conversation_id, "user", request.message)
        
        # Get conversation history for context
        history = await memory.get_conversation_history(conversation_id)
        
        # Process query through Crew.AI agents
        result = await orchestrator.process_query(
            query=request.message,
            conversation_history=history
        )
        
        # Save assistant response
        await memory.add_message(
            conversation_id,
            "assistant",
            result["response"],
            metadata={"citations": result.get("citations", [])}
        )
        
        return ChatResponse(
            response=result["response"],
            conversation_id=conversation_id,
            citations=result.get("citations", []),
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: ConversationCreate):
    """Create a new conversation"""
    try:
        memory = ConversationMemory()
        conversation = await memory.create_conversation(user_id=request.user_id)
        
        return ConversationResponse(
            conversation_id=conversation["conversation_id"],
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            message_count=0
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        memory = ConversationMemory()
        history = await memory.get_conversation_history(conversation_id)
        
        if not history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationHistoryResponse(
            conversation_id=conversation_id,
            messages=history["messages"],
            created_at=history["created_at"],
            updated_at=history["updated_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/documents/ingest", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    """Manually ingest a document"""
    try:
        from src.indexer.document_processor import DocumentProcessor
        from src.indexer.chunking_strategy import ChunkingStrategy
        from src.indexer.vector_store import VectorStore
        
        # Process document
        processor = DocumentProcessor()
        doc_data = processor.process_file(request.file_path)
        doc_data["metadata"].update(request.metadata)
        
        # Chunk document
        chunker = ChunkingStrategy()
        chunks = chunker.chunk_documents([doc_data])
        
        # Store in vector database
        vector_store = VectorStore()
        vector_store.add_documents(chunks)
        
        return DocumentIngestResponse(
            success=True,
            message=f"Successfully ingested document: {request.file_path}",
            chunks_created=len(chunks)
        )
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}", exc_info=True)
        return DocumentIngestResponse(
            success=False,
            message=f"Error ingesting document: {str(e)}"
        )


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIRequest):
    """OpenAI-compatible chat completions endpoint for OpenWebUI"""
    try:
        import uuid
        import time
        
        # Extract user message from messages
        user_message = None
        conversation_id = None
        
        for msg in request.messages:
            if msg.role == "user":
                user_message = msg.content
            elif msg.role == "system":
                # System messages can be used for conversation context
                pass
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Initialize orchestrator
        orchestrator = CrewOrchestrator()
        
        # Get or create conversation
        memory = ConversationMemory()
        conversation = await memory.create_conversation()
        conversation_id = conversation["conversation_id"]
        
        # Save user message
        await memory.add_message(conversation_id, "user", user_message)
        
        # Get conversation history for context
        history = await memory.get_conversation_history(conversation_id)
        
        # Process query through Crew.AI agents
        result = await orchestrator.process_query(
            query=user_message,
            conversation_history=history
        )
        
        # Save assistant response
        await memory.add_message(
            conversation_id,
            "assistant",
            result["response"],
            metadata={"citations": result.get("citations", [])}
        )
        
        # Format OpenAI-compatible response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        if request.stream:
            # Streaming response
            async def generate_stream():
                for chunk in result["response"].split():
                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk + " "},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Final chunk
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Non-streaming response
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["response"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(user_message.split()),
                    "completion_tokens": len(result["response"].split()),
                    "total_tokens": len(user_message.split()) + len(result["response"].split())
                }
            }
        
    except Exception as e:
        logger.error(f"Error in OpenAI chat completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

