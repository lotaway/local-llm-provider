import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent_runtime import AgentRuntime, AgentStatus, AgentResult
from agents.agent_base import BaseAgent

class MockLLM:
    pass

class MockPlanningAgent(BaseAgent):
    def execute(self, input_data, context, stream_callback=None):
        plan = [
            {"task_id": "task_1", "description": "First task", "dependencies": []},
            {"task_id": "task_2", "description": "Second task", "dependencies": ["task_1"]},
        ]
        context["current_plan"] = plan
        return AgentResult(
            status=AgentStatus.SUCCESS,
            data=plan[0],
            message="Plan created",
            next_agent="task_llm",  # Should go to task_llm normally, but we simplify
            metadata={"plan": plan}
        )

class MockTaskAgent(BaseAgent):
    def execute(self, input_data, context, stream_callback=None):
        # Simulate task execution
        return AgentResult(
            status=AgentStatus.SUCCESS,
            data="Task 1 result",
            message="Task 1 done",
            next_agent="verification"
        )

class MockVerificationAgent(BaseAgent):
    def execute(self, input_data, context, stream_callback=None):
        context.setdefault("completed_tasks", []).append("task_1")
        return AgentResult(
            status=AgentStatus.SUCCESS, # Triggers update
            data="Verified",
            message="Verification passed",
            next_agent="end" # Stop here for test
        )

def test_process_logging():
    print("Starting process logging test...")
    
    if os.path.exists("process.md"):
        os.remove("process.md")
        
    llm = MockLLM()
    runtime = AgentRuntime(llm, max_iterations=5)
    
    # Register mocks
    runtime.register_agent("planning", MockPlanningAgent(llm))
    runtime.register_agent("task_llm", MockTaskAgent(llm))
    runtime.register_agent("verification", MockVerificationAgent(llm))
    
    # Run planning step
    print("\nExecuting Planning Agent...")
    runtime.state.status = runtime.state.status.RUNNING
    runtime.state.current_agent = "planning"
    res1 = runtime.agents["planning"].execute(None, runtime.state.context)
    # Manually trigger the hook logic from _run_loop since we are unit testing or we can run _run_loop
    # Let's run _run_loop properly but we need to handle "end" agent
    
    # Re-setup for full loop run
    runtime = AgentRuntime(llm, max_iterations=3)
    runtime.register_agent("planning", MockPlanningAgent(llm))
    runtime.register_agent("task_llm", MockTaskAgent(llm))
    runtime.register_agent("verification", MockVerificationAgent(llm))
    
    # We need an agent that stops the loop or we count iterations
    class EndAgent(BaseAgent):
        def execute(self, input_data, context, stream_callback=None):
            return AgentResult(AgentStatus.COMPLETE, "Done", "Finished")
            
    runtime.register_agent("end", EndAgent(llm))
    
    print("\nRunning Runtime Loop...")
    runtime.execute("Start", start_agent="planning")
    
    
    if "process_checklist" in runtime.state.context:
        print("\nProcess checklist found in context!")
        print("-" * 20)
        content = runtime.state.context["process_checklist"]
        print(content)
        print("-" * 20)
        
        # New format checks
        if "- [x] First task" in content and "- [ ] Second task" in content:
            print("SUCCESS: Task 1 marked as complete, Task 2 pending.")
        else:
            print("FAILURE: Incorrect checklist state.")
            print(f"Content found:\n{content}")
    else:
        print("FAILURE: process_checklist not found in context.")
    
    # Cleanup file if it exists (should not exist now but good to check)
    if os.path.exists("process.md"):
         print("WARNING: process.md file was created, but should not have been.")

if __name__ == "__main__":
    test_process_logging()
