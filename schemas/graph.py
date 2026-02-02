from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime


class Entity(BaseModel):
    stable_id: str
    type: str
    canonical_name: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class Relation(BaseModel):
    from_entity_id: str
    to_entity_id: str
    relation_type: str
    confidence: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_doc_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class Evidence(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    entity_refs: List[str] = Field(default_factory=list)
    relation_refs: List[str] = Field(default_factory=list)


class LTMNode(BaseModel):
    topic: str
    version: int = 1
    conclusion: str
    conditions: List[str] = Field(default_factory=list)
    confidence: float = 0.5
    sources: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
