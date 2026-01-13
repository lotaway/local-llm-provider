
import asyncio
import sys
import os

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.error_handler_agent import ErrorHandlerAgent
from agents.agent_runtime import AgentRuntime
from agents.agent_base import AgentStatus

class MockLLM:
    async def chat_at_once(self, messages, **kwargs):
        return """
```json
{
    "root_cause_analysis": {
        "primary_cause": "LLM_TIMEOUT",
        "confidence": 0.98,
        "explanation": "Timeout detected in log signatures.",
        "supporting_evidence": ["Timeout error in stack trace"],
        "counter_evidence": []
    },
    "verification_step": {
        "hypothesis": "LLM service is currently slow",
        "tool_to_verify": "ping_llm_service",
        "is_verified": true
    },
    "proposed_action": {
        "action": "increase_timeout",
        "risk_score": 0.1,
        "blast_radius": "current_request",
        "rollback_plan": "revert timeout",
        "confidence_threshold_met": true
    },
    "decision": "AUTOMATIC"
}
```
"""
    def extract_after_think(self, text):
        return text

async def test_error_flow():
    mock_llm = MockLLM()
    handler = ErrorHandlerAgent(mock_llm)
    
    # Simulate an exception packet
    try:
        raise ValueError("LLM connection timed out")
    except Exception as e:
        input_data = {"exception": e, "agent_name": "test_agent"}
    
    context = {"iteration_count": 1}
    result = await handler.execute(input_data, context, {})
    
    print(f"Status: {result.status}")
    print(f"Message: {result.message}")
    print(f"Data: {result.data}")

if __name__ == "__main__":
    asyncio.run(test_error_flow())
