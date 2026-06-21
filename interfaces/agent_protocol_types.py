from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from schemas.agent_protocol import ToolSchema
from schemas.capability import Capability


@dataclass
class AgentInfo:
    name: str
    description: str
    supported_task_types: List[str]
    capabilities: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "supported_task_types": self.supported_task_types,
            "capabilities": self.capabilities,
        }


class AgentMetadataProvider(ABC):
    @abstractmethod
    def get_available_agents(self) -> List[AgentInfo]:
        pass
    
    @abstractmethod
    def get_available_tools(self) -> List[ToolSchema]:
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[Capability]:
        pass


class StreamEventConverter(ABC):
    @abstractmethod
    def convert_chunk(self, chunk: str, agent_name: str) -> Any:
        pass
    
    @abstractmethod
    def convert_tool_call(self, tool_name: str, arguments: Dict) -> Any:
        pass
    
    @abstractmethod
    def convert_status(self, status: str) -> Any:
        pass
    
    @abstractmethod
    def convert_error(self, error_code: Any, message: str) -> Any:
        pass