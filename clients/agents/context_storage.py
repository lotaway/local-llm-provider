"""Context Storage Backends for Agent Runtime

This module provides different storage backends for persisting agent context:
- MemoryContextStorage: In-memory storage (default)
- RedisContextStorage: Redis-based storage with configurable prefix
- PostgreSQLContextStorage: PostgreSQL-based storage with JSONB support
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextStorage(ABC):
    """Abstract base class for context storage backends"""
    
    @abstractmethod
    def save(self, session_id: str, state_data: Dict[str, Any]) -> bool:
        """Save context state for a session
        
        Args:
            session_id: Unique session identifier
            state_data: Serialized state data
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load context state for a session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            State data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete context state for a session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if context exists for a session
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if exists, False otherwise
        """
        pass


class MemoryContextStorage(ContextStorage):
    """In-memory context storage (default, no persistence)"""
    
    def __init__(self):
        self.storage: Dict[str, Dict[str, Any]] = {}
        logger.info("Initialized MemoryContextStorage")
    
    def save(self, session_id: str, state_data: Dict[str, Any]) -> bool:
        try:
            self.storage[session_id] = state_data
            logger.debug(f"Saved context for session {session_id} to memory")
            return True
        except Exception as e:
            logger.error(f"Failed to save context to memory: {e}")
            return False
    
    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self.storage.get(session_id)
        if data:
            logger.debug(f"Loaded context for session {session_id} from memory")
        else:
            logger.debug(f"No context found for session {session_id} in memory")
        return data
    
    def delete(self, session_id: str) -> bool:
        try:
            if session_id in self.storage:
                del self.storage[session_id]
                logger.debug(f"Deleted context for session {session_id} from memory")
            return True
        except Exception as e:
            logger.error(f"Failed to delete context from memory: {e}")
            return False
    
    def exists(self, session_id: str) -> bool:
        return session_id in self.storage


class RedisContextStorage(ContextStorage):
    """Redis-based context storage with configurable prefix"""
    
    def __init__(self, redis_url: str = None, prefix: str = None):
        """Initialize Redis storage
        
        Args:
            redis_url: Redis connection URL (default: from REDIS_URL env var)
            prefix: Key prefix for context keys (default: from REDIS_CONTEXT_PREFIX env var or "agent_context:")
        """
        try:
            import redis
        except ImportError:
            raise ImportError("redis package is required for RedisContextStorage. Install with: pip install redis>=5.0.0")
        
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.prefix = prefix or os.getenv("REDIS_CONTEXT_PREFIX", "agent_context:")
        
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info(f"Initialized RedisContextStorage with prefix '{self.prefix}'")
        except Exception as e:
            logger.error(f"Failed to connect to Redis at {self.redis_url}: {e}")
            raise
    
    def _make_key(self, session_id: str) -> str:
        """Create Redis key with prefix"""
        return f"{self.prefix}{session_id}"
    
    def save(self, session_id: str, state_data: Dict[str, Any]) -> bool:
        try:
            key = self._make_key(session_id)
            # Serialize to JSON
            json_data = json.dumps(state_data, ensure_ascii=False)
            # Save with optional TTL
            ttl = int(os.getenv("REDIS_CONTEXT_TTL", 0))
            if ttl > 0:
                self.client.setex(key, ttl, json_data)
                logger.debug(f"Saved context for session {session_id} to Redis with TTL {ttl}s")
            else:
                self.client.set(key, json_data)
                logger.debug(f"Saved context for session {session_id} to Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to save context to Redis: {e}")
            return False
    
    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            key = self._make_key(session_id)
            json_data = self.client.get(key)
            if json_data:
                logger.debug(f"Loaded context for session {session_id} from Redis")
                return json.loads(json_data)
            else:
                logger.debug(f"No context found for session {session_id} in Redis")
                return None
        except Exception as e:
            logger.error(f"Failed to load context from Redis: {e}")
            return None
    
    def delete(self, session_id: str) -> bool:
        try:
            key = self._make_key(session_id)
            self.client.delete(key)
            logger.debug(f"Deleted context for session {session_id} from Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to delete context from Redis: {e}")
            return False
    
    def exists(self, session_id: str) -> bool:
        try:
            key = self._make_key(session_id)
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Failed to check existence in Redis: {e}")
            return False


class PostgreSQLContextStorage(ContextStorage):
    """PostgreSQL-based context storage with JSONB support"""
    
    def __init__(self, postgres_url: str = None):
        """Initialize PostgreSQL storage
        
        Args:
            postgres_url: PostgreSQL connection URL (default: from POSTGRES_URL env var)
        """
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError("psycopg2 package is required for PostgreSQLContextStorage. Install with: pip install psycopg2-binary>=2.9.0")
        
        self.postgres_url = postgres_url or os.getenv("POSTGRES_URL")
        if not self.postgres_url:
            raise ValueError("POSTGRES_URL environment variable is required for PostgreSQLContextStorage")
        
        try:
            self.conn = psycopg2.connect(self.postgres_url)
            self.conn.autocommit = True
            logger.info("Initialized PostgreSQLContextStorage")
            
            # Ensure table exists
            self._create_table_if_not_exists()
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _create_table_if_not_exists(self):
        """Create agent_contexts table if it doesn't exist"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS agent_contexts (
                        session_id VARCHAR(255) PRIMARY KEY,
                        state_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_agent_contexts_updated_at 
                    ON agent_contexts(updated_at);
                """)
                logger.info("Ensured agent_contexts table exists")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    def save(self, session_id: str, state_data: Dict[str, Any]) -> bool:
        try:
            import psycopg2.extras
            
            with self.conn.cursor() as cur:
                # Upsert (INSERT ... ON CONFLICT UPDATE)
                cur.execute("""
                    INSERT INTO agent_contexts (session_id, state_data, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) 
                    DO UPDATE SET 
                        state_data = EXCLUDED.state_data,
                        updated_at = CURRENT_TIMESTAMP
                """, (session_id, psycopg2.extras.Json(state_data)))
                
                logger.debug(f"Saved context for session {session_id} to PostgreSQL")
                return True
        except Exception as e:
            logger.error(f"Failed to save context to PostgreSQL: {e}")
            return False
    
    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT state_data FROM agent_contexts 
                    WHERE session_id = %s
                """, (session_id,))
                
                row = cur.fetchone()
                if row:
                    logger.debug(f"Loaded context for session {session_id} from PostgreSQL")
                    return row[0]
                else:
                    logger.debug(f"No context found for session {session_id} in PostgreSQL")
                    return None
        except Exception as e:
            logger.error(f"Failed to load context from PostgreSQL: {e}")
            return None
    
    def delete(self, session_id: str) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM agent_contexts WHERE session_id = %s
                """, (session_id,))
                logger.debug(f"Deleted context for session {session_id} from PostgreSQL")
                return True
        except Exception as e:
            logger.error(f"Failed to delete context from PostgreSQL: {e}")
            return False
    
    def exists(self, session_id: str) -> bool:
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 1 FROM agent_contexts WHERE session_id = %s
                """, (session_id,))
                return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check existence in PostgreSQL: {e}")
            return False
    
    def cleanup_old_sessions(self, days: int = 7) -> int:
        """Clean up sessions older than specified days
        
        Args:
            days: Number of days to keep sessions
            
        Returns:
            Number of sessions deleted
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM agent_contexts 
                    WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '%s days'
                    RETURNING session_id
                """, (days,))
                deleted_count = cur.rowcount
                logger.info(f"Cleaned up {deleted_count} old sessions (older than {days} days)")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0


def create_context_storage(backend: str = None) -> ContextStorage:
    """Factory function to create context storage backend
    
    Args:
        backend: Storage backend type ('memory', 'redis', 'postgresql')
                 If None, reads from CONTEXT_STORAGE_BACKEND env var
    
    Returns:
        ContextStorage instance
    """
    backend = backend or os.getenv("CONTEXT_STORAGE_BACKEND", "memory").lower()
    
    logger.info(f"Creating context storage backend: {backend}")
    
    if backend == "memory":
        return MemoryContextStorage()
    elif backend == "redis":
        return RedisContextStorage()
    elif backend == "postgresql":
        return PostgreSQLContextStorage()
    else:
        logger.warning(f"Unknown storage backend '{backend}', falling back to memory")
        return MemoryContextStorage()
