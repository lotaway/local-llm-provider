"""Base Agent Class for Multi-Agent System"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status"""

    SUCCESS = "success"
    FAILURE = "failure"
    NEEDS_RETRY = "needs_retry"
    NEEDS_HUMAN = "needs_human"
    CONTINUE = "continue"
    COMPLETE = "complete"


@dataclass
class AgentResult:
    """Standard agent execution result"""

    status: AgentStatus
    data: Any
    message: str = ""
    metadata: Dict[str, Any] = None
    next_agent: Optional[str] = None
    private_data: Optional[Dict[str, Any]] = None
    thought_process: Optional[str] = None
    tool_calls: list = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.tool_calls is None:
            self.tool_calls = []


class BaseAgent(ABC):
    """Base class for all agents in the system"""

    def __init__(self, llm_model, name: str = None):
        """
        Initialize agent with LLM model instance

        Args:
            llm_model: LocalLLModel instance shared across all agents
            name: Agent name for logging
        """
        self.llm = llm_model
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"agent.{self.name}")

    @abstractmethod
    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        pass

    def _format_prompt(self, template: str, **kwargs) -> str:
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise

    async def _call_llm(
        self, messages: list[dict], stream_callback=None, **kwargs
    ) -> Dict[str, str]:
        if stream_callback:
            return await self._call_llm_stream(messages, stream_callback, **kwargs)
        
        raw_response = await self.llm.chat_at_once(messages, **kwargs)
        thought = self.llm.extract_thought(raw_response)
        response = self.llm.extract_after_think(raw_response)
        
        return {"response": response, "thought": thought}

    async def _call_llm_stream(
        self, messages: list[dict], stream_callback, **kwargs
    ) -> Dict[str, str]:
        full_response = ""
        async for chunk in self.llm.chat(messages, **kwargs):
            if isinstance(chunk, str) and chunk:
                full_response += chunk
                await self._notify_stream(stream_callback, chunk)

        return {
            "response": self.llm.extract_after_think(full_response),
            "thought": self.llm.extract_thought(full_response)
        }

    async def _notify_stream(self, callback, chunk):
        if not callback:
            return
        if asyncio.iscoroutinefunction(callback):
            await callback(chunk)
        else:
            callback(chunk)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        import json
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            response = json_match.group(0)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            try:
                return json.loads(response, strict=False)
            except json.JSONDecodeError:
                try:

                    def clean_string_value(match):
                        string_content = match.group(1)
                        string_content = string_content.replace("\n", "\\n")
                        string_content = string_content.replace("\r", "\\r")
                        string_content = string_content.replace("\t", "\\t")
                        return f'"{string_content}"'

                    cleaned_response = re.sub(
                        r'"([^"]*)"', clean_string_value, response
                    )
                    return json.loads(cleaned_response)
                except Exception:
                    sanitized_response = (
                        response[:500] + "..." if len(response) > 500 else response
                    )
                    sanitized_response = (
                        sanitized_response.replace("\n", "\\n")
                        .replace("\r", "\\r")
                        .replace("\t", "\\t")
                    )
                    self.logger.error(
                        f"Failed to parse JSON response: {e}\nResponse: {sanitized_response}"
                    )
                    raise ValueError(f"Failed to parse JSON response: {e}")

    def log_execution(self, input_data: Any, result: AgentResult):
        self.logger.info(
            f"Agent: {self.name} | Status: {result.status.value} | "
            f"Next: {result.next_agent or 'None'} | Message: {result.message}"
        )
        if result.metadata:
            self.logger.debug(f"  Metadata: {result.metadata}")
        if result.data:
            data_preview = (
                str(result.data)[:300] + "..."
                if len(str(result.data)) > 300
                else str(result.data)
            )
            self.logger.debug(f"  Data: {data_preview}")
