"""Conversation memory system using PostgreSQL"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """PostgreSQL-based conversation memory"""
    
    def __init__(self):
        """Initialize database connection"""
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    async def check_connection(self) -> bool:
        """Check database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False
    
    def _get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    async def create_conversation(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new conversation
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Conversation dictionary
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.now()
        
        try:
            with self._get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO conversations (conversation_id, created_at, updated_at)
                        VALUES (:conversation_id, :created_at, :updated_at)
                    """),
                    {
                        "conversation_id": conversation_id,
                        "created_at": now,
                        "updated_at": now
                    }
                )
                session.commit()
            
            logger.info(f"Created conversation: {conversation_id}")
            return {
                "conversation_id": conversation_id,
                "created_at": now,
                "updated_at": now
            }
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a message to a conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Message dictionary
        """
        import json
        
        message_id = str(uuid.uuid4())
        now = datetime.now()
        
        try:
            with self._get_session() as session:
                # Add message
                session.execute(
                    text("""
                        INSERT INTO messages (id, conversation_id, role, content, metadata, created_at)
                        VALUES (:id, :conversation_id, :role, :content, :metadata, :created_at)
                    """),
                    {
                        "id": message_id,
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content,
                        "metadata": json.dumps(metadata) if metadata else None,
                        "created_at": now
                    }
                )
                
                # Update conversation timestamp
                session.execute(
                    text("""
                        UPDATE conversations
                        SET updated_at = :updated_at
                        WHERE conversation_id = :conversation_id
                    """),
                    {
                        "conversation_id": conversation_id,
                        "updated_at": now
                    }
                )
                session.commit()
            
            logger.info(f"Added {role} message to conversation: {conversation_id}")
            return {
                "id": message_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "metadata": metadata,
                "created_at": now
            }
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            raise
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get conversation history
        
        Args:
            conversation_id: Conversation ID
            limit: Optional limit on number of messages
            
        Returns:
            Conversation history dictionary
        """
        import json
        
        try:
            with self._get_session() as session:
                # Get conversation
                conv_result = session.execute(
                    text("""
                        SELECT conversation_id, created_at, updated_at
                        FROM conversations
                        WHERE conversation_id = :conversation_id
                    """),
                    {"conversation_id": conversation_id}
                ).fetchone()
                
                if not conv_result:
                    return None
                
                # Get messages
                query = """
                    SELECT id, role, content, metadata, created_at
                    FROM messages
                    WHERE conversation_id = :conversation_id
                    ORDER BY created_at ASC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                messages_result = session.execute(
                    text(query),
                    {"conversation_id": conversation_id}
                ).fetchall()
                
                messages = []
                for msg in messages_result:
                    metadata = json.loads(msg.metadata) if msg.metadata else None
                    messages.append({
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at,
                        "metadata": metadata
                    })
                
                return {
                    "conversation_id": conv_result.conversation_id,
                    "created_at": conv_result.created_at,
                    "updated_at": conv_result.updated_at,
                    "messages": messages
                }
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            raise
    
    async def get_recent_messages(
        self,
        conversation_id: str,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages from a conversation
        
        Args:
            conversation_id: Conversation ID
            n: Number of recent messages
            
        Returns:
            List of recent messages
        """
        history = await self.get_conversation_history(conversation_id, limit=n)
        if history:
            return history["messages"]
        return []

