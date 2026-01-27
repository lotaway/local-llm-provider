from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class CapabilityKind(Enum):
    SKILL = "SKILL"
    MCP = "MCP"


@dataclass
class Capability:
    id: str
    kind: CapabilityKind
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    safety_level: str = "SAFE"
    permission: Optional[str] = None
    cost_hint: Optional[str] = None
    latency_hint: Optional[str] = None
    when_to_use: Optional[str] = None
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "safety_level": self.safety_level,
            "permission": self.permission,
            "cost_hint": self.cost_hint,
            "latency_hint": self.latency_hint,
            "when_to_use": self.when_to_use,
            "requires_approval": self.requires_approval,
        }
