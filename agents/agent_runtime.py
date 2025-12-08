"""Agent Runtime for managing agent execution flow"""

import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import json
from .agent_base import BaseAgent, AgentResult, AgentStatus
from .context_storage import ContextStorage, create_context_storage

logger = logging.getLogger(__name__)


class RuntimeStatus(Enum):
    """Runtime execution status"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_HUMAN = "waiting_human"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class RuntimeState:
    """Runtime state tracking"""
    status: RuntimeStatus = RuntimeStatus.RUNNING
    current_agent: Optional[str] = None
    iteration_count: int = 0
    iteration_count_round: int = 0
    max_iterations: int = 20
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Any = None
    error_message: str = ""
    
    def add_to_history(self, agent_name: str, result: AgentResult):
        """Add agent execution to history"""
        self.history.append({
            "agent": agent_name,
            "status": result.status.value,
            "message": result.message,
            "iteration": self.iteration_count,
            "data": result.data,
            "metadata": result.metadata
        })
    
    def get_context_summary(self) -> str:
        """Get summary of execution history for context"""
        summary_parts = []
        for entry in self.history[-5:]:  # Last 5 entries
            summary_parts.append(
                f"[{entry['iteration']}] {entry['agent']}: {entry['status']} - {entry['message']}"
            )
        return "\n".join(summary_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for storage"""
        return {
            "status": self.status.value,
            "current_agent": self.current_agent,
            "iteration_count": self.iteration_count,
            "iteration_count_round": self.iteration_count_round,
            "max_iterations": self.max_iterations,
            "context": self.context,
            "history": self.history,
            "final_result": self.final_result,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuntimeState':
        """Deserialize state from dictionary"""
        state = cls(
            max_iterations=data.get("max_iterations", 20)
        )
        state.status = RuntimeStatus(data.get("status", "running"))
        state.current_agent = data.get("current_agent")
        state.iteration_count = data.get("iteration_count", 0)
        state.iteration_count_round = data.get("iteration_count_round", 0)
        state.context = data.get("context", {})
        state.history = data.get("history", [])
        state.final_result = data.get("final_result")
        state.error_message = data.get("error_message", "")
        return state


class AgentRuntime:
    """Runtime environment for agent execution with loop and branch support"""
    
    def __init__(self, llm_model, max_iterations: int, context_storage: ContextStorage = None, session_id: str = None):
        """
        Initialize runtime
        
        Args:
            llm_model: Shared LocalLLModel instance
            max_iterations: Maximum execution iterations to prevent infinite loops
            context_storage: Optional context storage backend
            session_id: Optional session ID for context persistence
        """
        self.llm = llm_model
        self.agents: Dict[str, BaseAgent] = {}
        self.state = RuntimeState(max_iterations=max_iterations)
        self.logger = logging.getLogger("agent.runtime")
        
        # Context storage
        self.context_storage = context_storage
        self.session_id = session_id
        
        # Load existing state if session_id provided and storage available
        if self.session_id and self.context_storage:
            loaded_state = self._load_state()
            if loaded_state:
                self.state = loaded_state
                self.logger.info(f"Loaded existing state for session {self.session_id}")
        
        # Human intervention callback
        self.human_callback: Optional[callable] = None
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent with the runtime"""
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
    
    def set_human_callback(self, callback: callable):
        """Set callback for human intervention requests"""
        self.human_callback = callback
    
    def _save_state(self) -> bool:
        """Save current state to storage if available"""
        if not self.session_id or not self.context_storage:
            return False
        
        try:
            state_data = self.state.to_dict()
            success = self.context_storage.save(self.session_id, state_data)
            if success:
                self.logger.debug(f"Saved state for session {self.session_id}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            return False
    
    def _load_state(self) -> Optional[RuntimeState]:
        """Load state from storage if available"""
        if not self.session_id or not self.context_storage:
            return None
        
        try:
            state_data = self.context_storage.load(self.session_id)
            if state_data:
                return RuntimeState.from_dict(state_data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return None
    
    def execute(self, initial_input: Any, start_agent: str = "qa", stream_callback=None, initial_context: Dict[str, Any] = None) -> RuntimeState:
        """
        Execute agent workflow starting from specified agent
        
        Args:
            initial_input: Initial input (usually user query)
            start_agent: Name of starting agent (default: "qa")
            stream_callback: Optional callback for streaming events
            initial_context: Optional initial context dictionary
            
        Returns:
            Final RuntimeState with results
        """
        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting new agent workflow execution")
        self.logger.info(f"  Start agent: {start_agent}")
        self.logger.info(f"  Max iterations: {self.state.max_iterations}")
        input_preview = str(initial_input)[:200] + '...' if len(str(initial_input)) > 200 else str(initial_input)
        self.logger.info(f"  Initial input: {input_preview}")
        self.logger.info(f"=" * 80)
        
        self.state = RuntimeState(max_iterations=self.state.max_iterations)
        self.state.context["initial_input"] = initial_input
        if initial_context:
            self.state.context.update(initial_context)
        self.state.current_agent = start_agent
        
        result = self._run_loop(initial_input, stream_callback)
        
        # Save final state
        self._save_state()
        
        return result

    def resume(self, decision_data: Dict[str, Any], stream_callback=None) -> RuntimeState:
        """
        Resume execution after human intervention
        
        Args:
            decision_data: Data from human decision (approved, feedback, etc.)
            stream_callback: Optional callback for streaming events
            
        Returns:
            Updated RuntimeState
        """
        if self.state.status != RuntimeStatus.WAITING_HUMAN:
            raise ValueError(f"Cannot resume: Runtime is in {self.state.status.value} state, not waiting_human")
            
        if decision_data.get("approved", False):
            self.state.status = RuntimeStatus.RUNNING
            self.logger.info("Resuming execution after human approval")
            
            # Use provided data if available, otherwise use data from last history entry or context
            current_input = decision_data.get("data")
            
            # If feedback provided, add to context
            if decision_data.get("feedback"):
                self.state.context["human_feedback"] = decision_data["feedback"]
            
            result = self._run_loop(current_input, stream_callback)
            
            # Save state after resume
            self._save_state()
            
            return result
        else:
            self.state.status = RuntimeStatus.FAILED
            self.state.error_message = f"Human rejected operation: {decision_data.get('feedback', 'No reason provided')}"
            self.logger.info(self.state.error_message)
            return self.state

    def resume_after_max_iterations(self, stream_callback=None) -> RuntimeState:
        """
        Resume execution after reaching max_iterations
        User has decided to continue, so reset iteration_count and continue
        
        Args:
            stream_callback: Optional callback for streaming events
            
        Returns:
            Updated RuntimeState
        """
        if self.state.status != RuntimeStatus.MAX_ITERATIONS:
            raise ValueError(f"Cannot resume: Runtime is in {self.state.status.value} state, not max_iterations")
        
        # Reset iteration count for new round
        self.state.iteration_count = 0
        self.state.status = RuntimeStatus.RUNNING
        self.logger.info(f"Resuming execution after max_iterations (round {self.state.iteration_count_round})")
        
        # Get the last input from context or history
        current_input = None
        if self.state.history:
            last_entry = self.state.history[-1]
            current_input = last_entry.get("data")
        
        if current_input is None:
            current_input = self.state.context.get("initial_input")
        
        result = self._run_loop(current_input, stream_callback)
        
        # Save state after resume
        self._save_state()
        
        return result

    def _update_process_file(self, plan: List[Dict[str, Any]], completed_tasks: List[str]):
        """
        Update process.md with current execution status
        
        Args:
            plan: List of task definitions from planning agent
            completed_tasks: List of completed task IDs
        """
        try:
            original_query = self.state.context.get("original_query", "任务清单")
            # If query is too long, truncate it
            title = original_query[:50] + "..." if len(original_query) > 50 else original_query
            
            content = [f"# {title}"]
            
            for task in plan:
                task_id = task.get("task_id", "")
                description = task.get("description", "")
                
                is_completed = task_id in completed_tasks
                checkbox = "[x]" if is_completed else "[ ]"
                
                content.append(f"- {checkbox} {description}")
            
            # Store in context instead of writing to file
            process_checklist = "\n".join(content)
            self.state.context["process_checklist"] = process_checklist
            self.logger.info(f"\nExample process.md content stored in context:\n{process_checklist}")
                
        except Exception as e:
            self.logger.error(f"Failed to update process checklist: {e}")

    def _run_loop(self, current_input: Any, stream_callback=None) -> RuntimeState:
        """Internal execution loop"""
        self.logger.debug(f"Entering execution loop with status: {self.state.status.value}")
        
        while self.state.status == RuntimeStatus.RUNNING:
            # Check iteration limit
            if self.state.iteration_count >= self.state.max_iterations:
                self.state.status = RuntimeStatus.MAX_ITERATIONS
                self.state.iteration_count_round += 1
                self.state.error_message = f"Reached maximum iterations ({self.state.max_iterations}) in round {self.state.iteration_count_round}. User decision required to continue."
                self.logger.warning(self.state.error_message)
                self.logger.warning(f"Execution history: {self.state.get_context_summary()}")
                # Save state before pausing
                self._save_state()
                # Don't raise error, return control to user for decision
                break
            
            self.state.iteration_count += 1
            agent_name = self.state.current_agent
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Iteration {self.state.iteration_count}/{self.state.max_iterations} (Round {self.state.iteration_count_round})")
            self.logger.info(f"Current agent: {agent_name}")
            self.logger.info(f"{'='*60}")
            
            # Get agent
            if agent_name not in self.agents:
                self.state.status = RuntimeStatus.FAILED
                self.state.error_message = f"Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}"
                self.logger.error(self.state.error_message)
                break
            
            agent = self.agents[agent_name]
            
            # Log current input
            input_preview = str(current_input)[:300] + '...' if len(str(current_input)) > 300 else str(current_input)
            self.logger.debug(f"Input to {agent_name}: {input_preview}")
            
            # Emit agent start event
            if stream_callback:
                stream_callback({
                    "event_type": "agent_start",
                    "agent_name": agent_name,
                    "iteration": self.state.iteration_count
                })
            
            try:
                # Create a wrapper callback for LLM chunks
                def llm_chunk_callback(chunk: str):
                    if stream_callback:
                        stream_callback({
                            "event_type": "llm_chunk",
                            "agent_name": agent_name,
                            "chunk": chunk
                        })
                
                # Execute agent with streaming callback
                result = agent.execute(current_input, self.state.context, stream_callback=llm_chunk_callback)
                agent.log_execution(current_input, result)
                
                # Update state
                self.state.add_to_history(agent_name, result)
                
                # Emit agent complete event
                if stream_callback:
                    stream_callback({
                        "event_type": "agent_complete",
                        "agent_name": agent_name,
                        "status": result.status.value,
                        "next_agent": result.next_agent,
                        "message": result.message
                    })
                
                # Update process.md if critical agents completed
                if result.status == AgentStatus.SUCCESS:
                    should_update = False
                    
                    if agent_name == "planning":
                        # Planning agent completed, we should have a plan
                        # Try to get plan from metadata or context
                        plan = result.metadata.get("plan") or self.state.context.get("current_plan")
                        if plan:
                            should_update = True
                            
                    elif agent_name == "verification":
                        # Verification completed, a task might be done
                        should_update = True
                        
                    if should_update:
                        plan = self.state.context.get("current_plan", [])
                        completed_tasks = self.state.context.get("completed_tasks", [])
                        if plan:
                            self._update_process_file(plan, completed_tasks)
                
                # Handle result status
                if result.status == AgentStatus.COMPLETE:
                    self.state.status = RuntimeStatus.COMPLETED
                    self.state.final_result = result.data
                    self.logger.info(f"\n{'*'*60}")
                    self.logger.info(f"✓ Workflow completed successfully after {self.state.iteration_count} iterations")
                    self.logger.info(f"{'*'*60}")
                    # Save final state
                    self._save_state()
                    break
                
                elif result.status == AgentStatus.NEEDS_HUMAN:
                    self.state.status = RuntimeStatus.WAITING_HUMAN
                    
                    # Prepare for potential resume
                    self.state.current_agent = result.next_agent or "planning"
                    
                    if self.human_callback:
                        # Call human intervention callback
                        human_response = self.human_callback(result.data, self.state)
                        if human_response.get("approved", False):
                            # Continue with human approval
                            self.state.status = RuntimeStatus.RUNNING
                            current_input = human_response.get("data", result.data)
                        else:
                            # Human rejected, stop execution
                            self.state.error_message = "Human intervention rejected operation"
                            # Save state before breaking
                            self._save_state()
                            break
                    else:
                        self.logger.info("Pausing execution for human intervention")
                        # Save state before pausing
                        self._save_state()
                        # Break loop to return control to caller (e.g. API)
                        # The state is already set to WAITING_HUMAN
                        break
                
                elif result.status == AgentStatus.FAILURE:
                    self.state.status = RuntimeStatus.FAILED
                    self.state.error_message = result.message
                    self.logger.error(f"\n{'!'*60}")
                    self.logger.error(f"✗ Agent {agent_name} failed: {result.message}")
                    self.logger.error(f"  Iteration: {self.state.iteration_count}")
                    self.logger.error(f"  Context keys: {list(self.state.context.keys())}")
                    self.logger.error(f"  History length: {len(self.state.history)}")
                    if result.data:
                        self.logger.error(f"  Error data: {result.data}")
                    self.logger.error(f"{'!'*60}")
                    break
                
                elif result.status == AgentStatus.NEEDS_RETRY:
                    # Retry with same agent or update context for replanning
                    current_input = result.data
                    
                    # If this is MCP tool unavailable, ensure context has tool availability
                    if isinstance(result.data, dict) and result.data.get("error") == "tool_not_found":
                        # Update context with available MCP tools if not already present
                        if "available_mcp_tools" not in self.state.context:
                            self.state.context["available_mcp_tools"] = result.data.get("available_mcp_tools", [])
                    
                    self.logger.info(f"Retrying {agent_name}")
                
                elif result.status in [AgentStatus.SUCCESS, AgentStatus.CONTINUE]:
                    # Continue to next agent
                    if result.next_agent:
                        self.state.current_agent = result.next_agent
                        current_input = result.data
                    else:
                        self.state.status = RuntimeStatus.FAILED
                        self.state.error_message = "No next agent specified"
                        break
                
            except Exception as e:
                self.state.status = RuntimeStatus.FAILED
                self.state.error_message = f"Agent execution error: {str(e)}"
                self.logger.error(f"\n{'!'*60}")
                self.logger.error(f"✗ Exception during {agent_name} execution")
                self.logger.error(f"  Error: {str(e)}")
                self.logger.error(f"  Iteration: {self.state.iteration_count}")
                self.logger.error(f"{'!'*60}")
                self.logger.error("Full traceback:", exc_info=True)
                break
        
        return self.state
    
    def get_state(self) -> RuntimeState:
        """Get current runtime state"""
        return self.state
    
    def reset(self):
        """Reset runtime state"""
        self.state = RuntimeState(max_iterations=self.state.max_iterations)
        self.logger.info("Runtime state reset")
    
    @staticmethod
    def create_with_all_agents(llm_model, rag_instance=None, permission_manager=None, max_iterations: int = int(os.getenv("MAX_ITERATIONS", 100)), context_storage=None, session_id: str = None):
        """
        Factory method to create a fully initialized AgentRuntime with all standard agents
        
        Args:
            llm_model: LocalLLModel instance
            rag_instance: Optional LocalRAG instance (will be created if None)
            permission_manager: Optional PermissionManager instance
            max_iterations: Maximum iterations for runtime
            context_storage: Optional context storage backend
            session_id: Optional session ID for context persistence
            
        Returns:
            Fully initialized AgentRuntime
        """
        from .qa_agent import QAAgent
        from .planning_agent import PlanningAgent
        from .router_agent import RouterAgent
        from .verification_agent import VerificationAgent
        from .risk_agent import RiskAgent, RiskLevel
        from .task_agents.llm_agent import LLMTaskAgent
        from .task_agents.rag_agent import RAGTaskAgent
        from .task_agents.mcp_agent import MCPTaskAgent
        
        # Initialize RAG if needed
        if rag_instance is None:
            from rag import LocalRAG
            data_path = os.getenv("DATA_PATH", "./docs")
            rag_instance = LocalRAG(llm_model, data_path=data_path)
        
        # Initialize permission manager if needed
        if permission_manager is None:
            from permission_manager import PermissionManager, SafetyLevel
            permission_manager = PermissionManager(human_approval_threshold=SafetyLevel.HIGH)
        
        # Create runtime
        runtime = AgentRuntime(llm_model, max_iterations=max_iterations, context_storage=context_storage, session_id=session_id)
        
        # Register core agents
        runtime.register_agent("qa", QAAgent(llm_model))
        runtime.register_agent("planning", PlanningAgent(llm_model))
        runtime.register_agent("router", RouterAgent(llm_model))
        runtime.register_agent("verification", VerificationAgent(llm_model))
        runtime.register_agent("risk", RiskAgent(llm_model, risk_threshold=RiskLevel.HIGH))
        
        # Register task agents
        runtime.register_agent("task_llm", LLMTaskAgent(llm_model))
        runtime.register_agent("task_rag", RAGTaskAgent(llm_model, rag_instance=rag_instance))
        
        # Register MCP agent with permission manager
        mcp_agent = MCPTaskAgent(llm_model)
        mcp_agent.permission_manager = permission_manager
        runtime.register_agent("task_mcp", mcp_agent)
        
        # Populate context with available MCP tools for planning agent
        runtime.state.context["available_mcp_tools"] = mcp_agent.get_available_tools()
        
        logger.info("Created AgentRuntime with all standard agents")
        
        return runtime
