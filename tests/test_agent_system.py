"""Simple test script for agent system"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_provider import LocalLLModel
from agents import AgentRuntime
from agents.qa_agent import QAAgent
from agents.planning_agent import PlanningAgent
from agents.router_agent import RouterAgent
from agents.verification_agent import VerificationAgent
from agents.risk_agent import RiskAgent, RiskLevel
from agents.task_agents.llm_agent import LLMTaskAgent
from agents.task_agents.rag_agent import RAGTaskAgent
from agents.task_agents.mcp_agent import MCPTaskAgent
from rag import LocalRAG


def test_agent_system():
    """Test the agent system with a simple query"""
    print("Initializing agent system...")
    
    # Initialize LLM
    llm = LocalLLModel()
    
    # Initialize RAG
    data_path = os.getenv("DATA_PATH", "./docs")
    rag = LocalRAG(llm, data_path=data_path, use_hybrid_search=True, use_reranking=True)
    
    # Create runtime
    runtime = AgentRuntime(llm, max_iterations=10)
    
    # Register agents
    runtime.register_agent("qa", QAAgent(llm))
    runtime.register_agent("planning", PlanningAgent(llm))
    runtime.register_agent("router", RouterAgent(llm))
    runtime.register_agent("verification", VerificationAgent(llm))
    runtime.register_agent("risk", RiskAgent(llm, risk_threshold=RiskLevel.HIGH))
    
    # Register task agents
    runtime.register_agent("task_llm", LLMTaskAgent(llm))
    runtime.register_agent("task_rag", RAGTaskAgent(llm, rag_instance=rag))
    runtime.register_agent("task_mcp", MCPTaskAgent(llm))
    
    print("Agent system initialized successfully!")
    print("\nRegistered agents:")
    for agent_name in runtime.agents.keys():
        print(f"  - {agent_name}")
    
    # Test query
    test_query = "What is 2 + 2?"
    print(f"\n{'='*60}")
    print(f"Test Query: {test_query}")
    print(f"{'='*60}\n")
    
    try:
        state = runtime.execute(test_query, start_agent="qa")
        
        print(f"\nExecution completed!")
        print(f"Status: {state.status.value}")
        print(f"Iterations: {state.iteration_count}")
        print(f"\nFinal Result:")
        print(f"{state.final_result}")
        
        if state.error_message:
            print(f"\nError: {state.error_message}")
        
        print(f"\n{'='*60}")
        print("Execution History:")
        print(f"{'='*60}")
        for i, entry in enumerate(state.history, 1):
            print(f"{i}. [{entry['iteration']}] {entry['agent']}: {entry['status']}")
            print(f"   Message: {entry['message']}")
        
        return state.status.value == "completed"
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_agent_system()
    sys.exit(0 if success else 1)
