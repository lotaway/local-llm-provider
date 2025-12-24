import asyncio
from typing import Any, Dict
import sys
import os


# Mock LLM model
class MockLLM:
    async def chat_at_once(self, messages, **kwargs):
        return "Mock response"

    def extract_after_think(self, text):
        return text


# Add parent dir to path
sys.path.append(os.path.dirname(os.getcwd()))

from agents.agent_base import BaseAgent, AgentResult, AgentStatus
from agents.agent_runtime import AgentRuntime


class PrivateContextAgent(BaseAgent):
    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        # Check if we have something in private context
        count = private_context.get("count", 0)
        count += 1

        # Set shared context
        context["shared_key"] = "shared_value"

        # Determine next agent
        next_agent = None
        status = AgentStatus.SUCCESS
        if count < 3:
            next_agent = self.name
        else:
            status = AgentStatus.COMPLETE

        return AgentResult(
            status=status,
            data=f"Count is {count}",
            next_agent=next_agent,
            private_data={"count": count},
        )


async def test_private_context():
    llm = MockLLM()
    runtime = AgentRuntime(llm, max_iterations=10)

    agent = PrivateContextAgent(llm, name="test_agent")
    runtime.register_agent("test_agent", agent)

    print("Starting execution...")
    state = await runtime.execute("start", start_agent="test_agent")

    print(f"Status: {state.status}")
    print(f"Final Result: {state.final_result}")
    print(f"Shared Context: {state.context}")
    print(f"Private Contexts: {state.private_contexts}")

    assert state.private_contexts["test_agent"]["count"] == 3
    assert state.context["shared_key"] == "shared_value"
    print("Verification SUCCESS!")


if __name__ == "__main__":
    asyncio.run(test_private_context())
