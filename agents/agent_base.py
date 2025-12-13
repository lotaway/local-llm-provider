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

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


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
        self, input_data: Any, context: Dict[str, Any], stream_callback=None
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
    ) -> str:
        import time

        self.logger.debug(f"LLM call started with {len(messages)} messages")
        for i, msg in enumerate(messages):
            content_preview = (
                msg.get("content", "")[:200] + "..."
                if len(msg.get("content", "")) > 200
                else msg.get("content", "")
            )
            self.logger.debug(f"  Message {i} [{msg.get('role')}]: {content_preview}")

        start_time = time.time()
        try:
            if stream_callback:
                # Use streaming mode
                response = await self._call_llm_stream(
                    messages, stream_callback, **kwargs
                )
            else:
                # Use non-streaming mode
                response = await self.llm.chat_at_once(messages, **kwargs)
                response = self.llm.extract_after_think(response)

            elapsed = time.time() - start_time
            response_preview = (
                response[:200] + "..." if len(response) > 200 else response
            )
            self.logger.debug(
                f"LLM call completed in {elapsed:.2f}s, response length: {len(response)} chars"
            )
            self.logger.debug(f"  Response preview: {response_preview}")

            return response
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"LLM call failed after {elapsed:.2f}s: {e}", exc_info=True
            )
            raise

    async def _call_llm_stream(
        self, messages: list[dict], stream_callback, **kwargs
    ) -> str:
        try:
            self.logger.debug("Starting LLM streaming call")
            full_response = ""
            chunk_count = 0

            import asyncio

            async for chunk in self.llm.chat(messages, **kwargs):
                if isinstance(chunk, int):
                    continue

                if chunk:
                    full_response += chunk
                    chunk_count += 1
                    if stream_callback:
                        if asyncio.iscoroutinefunction(stream_callback):
                            await stream_callback(chunk)
                        else:
                            stream_callback(chunk)

            self.logger.debug(
                f"LLM streaming completed: {chunk_count} chunks, {len(full_response)} total chars"
            )
            return self.llm.extract_after_think(full_response)
        except Exception as e:
            self.logger.error(f"LLM streaming call failed: {e}", exc_info=True)
            raise

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
