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
    def execute(self, input_data: Any, context: Dict[str, Any], stream_callback=None) -> AgentResult:
        """
        Execute agent logic
        
        Args:
            input_data: Input data for this agent
            context: Shared context from runtime (history, state, etc.)
            stream_callback: Optional callback for streaming LLM outputs
            
        Returns:
            AgentResult with status, data, and next steps
        """
        pass
    
    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format prompt template with variables"""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise
    
    def _call_llm(self, messages: list[dict], stream_callback=None, **kwargs) -> str:
        """
        Call LLM with messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream_callback: Optional callback function to receive streaming chunks
            **kwargs: Additional generation parameters
            
        Returns:
            LLM response text
        """
        import time
        
        # Log input (truncated for readability)
        self.logger.debug(f"LLM call started with {len(messages)} messages")
        for i, msg in enumerate(messages):
            content_preview = msg.get('content', '')[:200] + '...' if len(msg.get('content', '')) > 200 else msg.get('content', '')
            self.logger.debug(f"  Message {i} [{msg.get('role')}]: {content_preview}")
        
        start_time = time.time()
        try:
            if stream_callback:
                # Use streaming mode
                response = self._call_llm_stream(messages, stream_callback, **kwargs)
            else:
                # Use non-streaming mode
                response = self.llm.chat_at_once(messages, **kwargs)
                response = self.llm.extract_after_think(response)
            
            elapsed = time.time() - start_time
            response_preview = response[:200] + '...' if len(response) > 200 else response
            self.logger.debug(f"LLM call completed in {elapsed:.2f}s, response length: {len(response)} chars")
            self.logger.debug(f"  Response preview: {response_preview}")
            
            return response
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"LLM call failed after {elapsed:.2f}s: {e}", exc_info=True)
            raise
    
    def _call_llm_stream(self, messages: list[dict], stream_callback, **kwargs) -> str:
        """
        Call LLM with streaming support
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream_callback: Callback function to receive streaming chunks
            **kwargs: Additional generation parameters
            
        Returns:
            Complete LLM response text
        """
        try:
            self.logger.debug("Starting LLM streaming call")
            streamer = self.llm.chat(messages)
            full_response = ""
            chunk_count = 0
            
            for chunk in streamer:
                if chunk:
                    full_response += chunk
                    chunk_count += 1
                    # Call the callback with the chunk
                    if stream_callback:
                        stream_callback(chunk)
            
            self.logger.debug(f"LLM streaming completed: {chunk_count} chunks, {len(full_response)} total chars")
            return self.llm.extract_after_think(full_response)
        except Exception as e:
            self.logger.error(f"LLM streaming call failed: {e}", exc_info=True)
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling control characters"""
        import json
        import re
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)
        
        try:
            # First attempt: try parsing as-is
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Second attempt: try with strict=False to be more lenient
            try:
                return json.loads(response, strict=False)
            except json.JSONDecodeError:
                # Third attempt: manually clean control characters in string values
                try:
                    # Replace common control characters in JSON string values
                    # This regex finds string values and replaces control chars within them
                    def clean_string_value(match):
                        string_content = match.group(1)
                        # Replace control characters with escaped versions
                        string_content = string_content.replace('\n', '\\n')
                        string_content = string_content.replace('\r', '\\r')
                        string_content = string_content.replace('\t', '\\t')
                        return f'"{string_content}"'
                    
                    # Find all string values in JSON and clean them
                    cleaned_response = re.sub(r'"([^"]*)"', clean_string_value, response)
                    return json.loads(cleaned_response)
                except Exception:
                    # If all attempts fail, log error with sanitized response
                    # Truncate response for logging to avoid huge error messages
                    sanitized_response = response[:500] + "..." if len(response) > 500 else response
                    # Replace control characters for readable error message
                    sanitized_response = sanitized_response.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    self.logger.error(f"Failed to parse JSON response: {e}\nResponse: {sanitized_response}")
                    raise ValueError(f"Failed to parse JSON response: {e}")

    
    def log_execution(self, input_data: Any, result: AgentResult):
        """Log agent execution for debugging"""
        # Log basic execution info
        self.logger.info(
            f"Agent: {self.name} | Status: {result.status.value} | "
            f"Next: {result.next_agent or 'None'} | Message: {result.message}"
        )
        
        # Log detailed info at debug level
        if result.metadata:
            self.logger.debug(f"  Metadata: {result.metadata}")
        if result.data:
            data_preview = str(result.data)[:300] + '...' if len(str(result.data)) > 300 else str(result.data)
            self.logger.debug(f"  Data: {data_preview}")
