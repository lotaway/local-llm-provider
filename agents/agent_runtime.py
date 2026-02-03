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
from .error_handler_agent import ErrorHandlerAgent

logger = logging.getLogger(__name__)


class RuntimeStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_HUMAN = "waiting_human"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class RuntimeState:
    status: RuntimeStatus = RuntimeStatus.RUNNING
    current_agent: Optional[str] = None
    iteration_count: int = 0
    iteration_count_round: int = 0
    max_iterations: int = 20
    context: Dict[str, Any] = field(default_factory=dict)
    private_contexts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Any = None
    final_meta: Any = None
    error_message: str = ""

    def add_to_history(self, agent_name: str, result: AgentResult):
        self.history.append(
            {
                "agent": agent_name,
                "status": result.status.value,
                "message": result.message,
                "iteration": self.iteration_count,
                "data": result.data,
                "metadata": result.metadata,
            }
        )

    def get_context_summary(self) -> str:
        summary_parts = []
        for entry in self.history[-5:]:
            summary_parts.append(
                f"[{entry['iteration']}] {entry['agent']}: {entry['status']} - {entry['message']}"
            )
        return "\n".join(summary_parts)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "status": self.status.value,
            "current_agent": self.current_agent,
            "iteration_count": self.iteration_count,
            "iteration_count_round": self.iteration_count_round,
            "max_iterations": self.max_iterations,
            "context": self.context,
            "private_contexts": self.private_contexts,
            "history": self.history,
            "final_result": self.final_result,
            "final_meta": self.final_meta,
            "error_message": self.error_message,
        }
        if self.status == RuntimeStatus.MAX_ITERATIONS:
            result["decision_data"] = {
                "reason": "max_iterations_reached",
                "message": self.error_message,
                "max_iterations": self.max_iterations,
                "iteration_count_round": self.iteration_count_round,
            }
        elif self.status == RuntimeStatus.WAITING_HUMAN:
            decision_data = {}
            if self.history:
                last_entry = self.history[-1]
                if last_entry.get("data"):
                    decision_data = last_entry["data"].copy()
            if "reason" not in decision_data:
                decision_data["reason"] = self.error_message or "waiting_human"
            if "message" not in decision_data:
                decision_data["message"] = self.error_message

            result["decision_data"] = decision_data

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeState":
        state = cls(max_iterations=data.get("max_iterations", 20))
        state.status = RuntimeStatus(data.get("status", "running"))
        state.current_agent = data.get("current_agent")
        state.iteration_count = data.get("iteration_count", 0)
        state.iteration_count_round = data.get("iteration_count_round", 0)
        state.context = data.get("context", {})
        state.private_contexts = data.get("private_contexts", {})
        state.history = data.get("history", [])
        state.final_result = data.get("final_result")
        state.final_meta = data.get("final_meta")
        state.error_message = data.get("error_message", "")
        return state


class AgentRuntime:
    def __init__(
        self,
        llm_model,
        max_iterations: int,
        context_storage: ContextStorage = None,
        session_id: str = None,
        feedback_judge=None,
        evolution_dispatcher=None,
    ):
        self.llm = llm_model
        self.agents: Dict[str, BaseAgent] = {}
        self.state = RuntimeState(max_iterations=max_iterations)
        self.logger = logging.getLogger("agent.runtime")
        self.context_storage = context_storage
        self.session_id = session_id
        self.feedback_judge = feedback_judge
        self.evolution_dispatcher = evolution_dispatcher
        if self.session_id and self.context_storage:
            loaded_state = self._load_state()
            if loaded_state:
                self.state = loaded_state
                self.logger.info(f"Loaded existing state for session {self.session_id}")
        self.human_callback: Optional[callable] = None

    def register_agent(self, name: str, agent: BaseAgent):
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")

    def set_human_callback(self, callback: callable):
        self.human_callback = callback

    def _save_state(self) -> bool:
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

    async def execute(
        self,
        initial_input: Any,
        start_agent: str = "qa",
        stream_callback=None,
        initial_context: Dict[str, Any] = None,
    ) -> RuntimeState:
        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting new agent workflow execution")
        self.logger.info(f"  Start agent: {start_agent}")
        self.logger.info(f"  Max iterations: {self.state.max_iterations}")
        input_preview = (
            str(initial_input)[:200] + "..."
            if len(str(initial_input)) > 200
            else str(initial_input)
        )
        self.logger.info(f"  Initial input: {input_preview}")
        self.logger.info(f"=" * 80)

        self.state = RuntimeState(max_iterations=self.state.max_iterations)
        self.state.context["initial_input"] = initial_input
        if initial_context:
            self.state.context.update(initial_context)
        self.state.current_agent = start_agent

        result = await self._run_loop(initial_input, stream_callback)

        # Save final state
        self._save_state()

        return result

    async def resume(
        self, decision_data: Dict[str, Any], stream_callback=None
    ) -> RuntimeState:
        """
        Resume execution after human intervention

        Args:
            decision_data: Data from human decision (approved, feedback, etc.)
            stream_callback: Optional callback for streaming events

        Returns:
            Updated RuntimeState
        """
        if self.state.status != RuntimeStatus.WAITING_HUMAN:
            raise ValueError(
                f"Cannot resume: Runtime is in {self.state.status.value} state, not waiting_human"
            )

        if decision_data.get("approved", False):
            self.state.status = RuntimeStatus.RUNNING
            self.logger.info("Resuming execution after human approval")

            # Use provided data if available, otherwise use data from last history entry or context
            current_input = decision_data.get("data")

            # If feedback provided, add to context
            if decision_data.get("feedback"):
                self.state.context["human_feedback"] = decision_data["feedback"]

            result = await self._run_loop(current_input, stream_callback)

            # Save state after resume
            self._save_state()

            return result
        else:
            self.state.status = RuntimeStatus.FAILED
            self.state.error_message = f"Human rejected operation: {decision_data.get('feedback', 'No reason provided')}"
            self.logger.info(self.state.error_message)
            return self.state

    async def resume_after_max_iterations(self, stream_callback=None) -> RuntimeState:
        if self.state.status != RuntimeStatus.MAX_ITERATIONS:
            raise ValueError(
                f"Cannot resume: Runtime is in {self.state.status.value} state, not max_iterations"
            )

        # Reset iteration count for new round
        self.state.iteration_count = 0
        self.state.status = RuntimeStatus.RUNNING
        self.logger.info(
            f"Resuming execution after max_iterations (round {self.state.iteration_count_round})"
        )

        # Get the last input from context or history
        current_input = None
        if self.state.history:
            last_entry = self.state.history[-1]
            current_input = last_entry.get("data")

        if current_input is None:
            current_input = self.state.context.get("initial_input")

        result = await self._run_loop(current_input, stream_callback)

        # Save state after resume
        self._save_state()

        return result

    def _update_process_file(
        self, plan: List[Dict[str, Any]], completed_tasks: List[str]
    ):
        try:
            original_query = self.state.context.get("original_query", "任务清单")
            title = (
                original_query[:50] + "..."
                if len(original_query) > 50
                else original_query
            )

            content = [f"# {title}"]

            for task in plan:
                task_id = task.get("task_id", "")
                description = task.get("description", "")

                is_completed = task_id in completed_tasks
                checkbox = "[x]" if is_completed else "[ ]"

                content.append(f"- {checkbox} {description}")
            process_checklist = "\n".join(content)
            self.state.context["process_checklist"] = process_checklist
            self.logger.info(
                f"\nExample process.md content stored in context:\n{process_checklist}"
            )

        except Exception as e:
            self.logger.error(f"Failed to update process checklist: {e}")

    async def _run_loop(self, current_input: Any, stream_callback=None) -> RuntimeState:
        self.logger.debug(
            f"Entering execution loop with status: {self.state.status.value}"
        )
        import asyncio

        while self.state.status == RuntimeStatus.RUNNING:
            if self.state.iteration_count >= self.state.max_iterations:
                self.state.status = RuntimeStatus.MAX_ITERATIONS
                self.state.iteration_count_round += 1
                self.state.error_message = f"Reached maximum iterations ({self.state.max_iterations}) in round {self.state.iteration_count_round}. User decision required to continue."
                self.logger.warning(self.state.error_message)
                self.logger.warning(
                    f"Execution history: {self.state.get_context_summary()}"
                )
                self._save_state()
                break

            self.state.iteration_count += 1
            agent_name = self.state.current_agent

            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(
                f"Iteration {self.state.iteration_count}/{self.state.max_iterations} (Round {self.state.iteration_count_round})"
            )
            self.logger.info(f"Current agent: {agent_name}")
            self.logger.info(f"{'=' * 60}")

            # Get agent
            if agent_name not in self.agents:
                self.state.status = RuntimeStatus.FAILED
                self.state.error_message = f"Agent '{agent_name}' not found. Available agents: {list(self.agents.keys())}"
                self.logger.error(self.state.error_message)
                break

            agent = self.agents[agent_name]
            input_preview = (
                str(current_input)[:300] + "..."
                if len(str(current_input)) > 300
                else str(current_input)
            )
            self.logger.debug(f"Input to {agent_name}: {input_preview}")
            if stream_callback:
                evt = {
                    "event_type": "agent_start",
                    "agent_name": agent_name,
                    "iteration": self.state.iteration_count,
                }
                if asyncio.iscoroutinefunction(stream_callback):
                    await stream_callback(evt)
                else:
                    stream_callback(evt)

            try:

                async def llm_chunk_callback_async(chunk: str):
                    if stream_callback:
                        evt = {
                            "event_type": "llm_chunk",
                            "agent_name": agent_name,
                            "chunk": chunk,
                        }
                        if asyncio.iscoroutinefunction(stream_callback):
                            await stream_callback(evt)
                        else:
                            stream_callback(evt)

                def llm_chunk_callback_sync(chunk: str):
                    if stream_callback:
                        evt = {
                            "event_type": "llm_chunk",
                            "agent_name": agent_name,
                            "chunk": chunk,
                        }
                        if asyncio.iscoroutinefunction(stream_callback):
                            pass
                        else:
                            stream_callback(evt)

                # Get private context for this agent
                private_context = self.state.private_contexts.get(agent_name, {})

                result = await agent.execute(
                    current_input,
                    self.state.context,
                    private_context,
                    stream_callback=llm_chunk_callback_async,
                )

                # Update private context if agent returned any
                if result.private_data:
                    if agent_name not in self.state.private_contexts:
                        self.state.private_contexts[agent_name] = {}
                    self.state.private_contexts[agent_name].update(result.private_data)

                agent.log_execution(current_input, result)
                self.state.add_to_history(agent_name, result)
                if stream_callback:
                    evt = {
                        "event_type": "agent_complete",
                        "agent_name": agent_name,
                        "status": result.status.value,
                        "next_agent": result.next_agent,
                        "message": result.message,
                    }
                    if asyncio.iscoroutinefunction(stream_callback):
                        await stream_callback(evt)
                    else:
                        stream_callback(evt)
                if result.status == AgentStatus.SUCCESS:
                    should_update = False
                    if agent_name == "planning":
                        plan = result.metadata.get("plan") or self.state.context.get(
                            "current_plan"
                        )
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
                    self.state.final_meta = self._build_agent_core_meta()
                    self.state.context["agent_core_meta"] = self.state.final_meta
                    await self._run_feedback_pipeline(
                        user_input=self._get_feedback_input(),
                        agent_response=result.data,
                        context_window=self._build_context_window(),
                    )
                    self.logger.info(f"\n{'*' * 60}")
                    self.logger.info(
                        f"✓ Workflow completed successfully after {self.state.iteration_count} iterations"
                    )
                    self.logger.info(f"{'*' * 60}")
                    # Save final state
                    self._save_state()
                    break

                elif result.status == AgentStatus.NEEDS_HUMAN:
                    self.state.status = RuntimeStatus.WAITING_HUMAN

                    # Prepare for potential resume
                    self.state.current_agent = result.next_agent or "planning"

                    if self.human_callback:
                        # Call human intervention callback
                        # Callback might be async? Usually callback is simple hook.
                        # If we execute async, we might want async callback.
                        if asyncio.iscoroutinefunction(self.human_callback):
                            human_response = await self.human_callback(
                                result.data, self.state
                            )
                        else:
                            human_response = self.human_callback(
                                result.data, self.state
                            )

                        if human_response.get("approved", False):
                            self.state.status = RuntimeStatus.RUNNING
                            current_input = human_response.get("data", result.data)
                        else:
                            self.state.error_message = (
                                "Human intervention rejected operation"
                            )
                            self._save_state()
                            break
                    else:
                        self.logger.info("Pausing execution for human intervention")
                        self._save_state()
                        break

                elif result.status == AgentStatus.FAILURE:
                    self.state.status = RuntimeStatus.FAILED
                    self.state.error_message = result.message
                    self.logger.error(f"\n{'!' * 60}")
                    self.logger.error(f"✗ Agent {agent_name} failed: {result.message}")
                    self.logger.error(f"  Iteration: {self.state.iteration_count}")
                    self.logger.error(
                        f"  Context keys: {list(self.state.context.keys())}"
                    )
                    self.logger.error(f"  History length: {len(self.state.history)}")
                    if result.data:
                        self.logger.error(f"  Error data: {result.data}")
                    self.logger.error(f"{'!' * 60}")
                    break

                elif result.status == AgentStatus.NEEDS_RETRY:
                    current_input = result.data
                    if (
                        isinstance(result.data, dict)
                        and result.data.get("error") == "tool_not_found"
                    ):
                        if "available_mcp_tools" not in self.state.context:
                            self.state.context["available_mcp_tools"] = result.data.get(
                                "available_mcp_tools", []
                            )

                    self.logger.info(f"Retrying {agent_name}")

                elif result.status in [AgentStatus.SUCCESS, AgentStatus.CONTINUE]:
                    if result.next_agent:
                        self.state.current_agent = result.next_agent
                        current_input = result.data
                    else:
                        self.state.status = RuntimeStatus.FAILED
                        self.state.error_message = "No next agent specified"
                        break

            except Exception as e:
                if "error_handler" in self.agents and agent_name != "error_handler":
                    self.logger.warning(
                        f"Exception in agent {agent_name}, routing to error_handler: {e}"
                    )
                    self.state.current_agent = "error_handler"
                    current_input = {"exception": e, "agent_name": agent_name}
                    continue

                self.state.status = RuntimeStatus.FAILED
                self.state.error_message = f"Agent execution error: {str(e)}"
                self.logger.error(f"\n{'!' * 60}")
                self.logger.error(f"✗ Exception during {agent_name} execution")
                self.logger.error(f"  Error: {str(e)}")
                self.logger.error(f"  Iteration: {self.state.iteration_count}")
                self.logger.error(f"{'!' * 60}")
                self.logger.error("Full traceback:", exc_info=True)
                break

        return self.state

    def get_state(self) -> RuntimeState:
        return self.state

    def reset(self):
        self.state = RuntimeState(max_iterations=self.state.max_iterations)
        self.logger.info("Runtime state reset")

    async def handle_decision(
        self, approved: bool, feedback: str = "", data: Dict[str, Any] = None
    ):
        decision_data = data or {}
        decision_data["approved"] = approved
        if feedback:
            decision_data["feedback"] = feedback
        if self.state.status == RuntimeStatus.MAX_ITERATIONS:
            return await self.resume_after_max_iterations()
        return await self.resume(decision_data)

    def _build_agent_core_meta(self) -> Dict[str, Any]:
        reasoning_path = " -> ".join(
            [entry.get("agent", "") for entry in self.state.history]
        )
        skills_used = list(dict.fromkeys(self.state.context.get("skills_used", [])))
        verifications = self.state.context.get("verifications", [])
        confidence = verifications[-1].get("confidence", 0.0) if verifications else 0.0
        execution_log = [
            {
                "agent": entry.get("agent"),
                "status": entry.get("status"),
                "message": entry.get("message"),
            }
            for entry in self.state.history
        ]
        return {
            "reasoning_path": reasoning_path,
            "skills_used": skills_used,
            "confidence": confidence,
            "execution_log": execution_log,
        }

    def _build_context_window(self) -> list:
        window = []
        for entry in self.state.history[-5:]:
            window.append(
                f"{entry.get('agent')}: {entry.get('message')} ({entry.get('status')})"
            )
        return window

    def _get_feedback_input(self) -> str:
        return self.state.context.get("human_feedback") or self.state.context.get("original_query", "")

    async def _run_feedback_pipeline(self, user_input: str, agent_response: Any, context_window: list):
        if not self.feedback_judge or not self.evolution_dispatcher:
            return
        try:
            signal = self.feedback_judge.evaluate(
                user_input=user_input,
                agent_response=str(agent_response),
                context_window=context_window,
            )
            self.evolution_dispatcher.enqueue(signal)
            self.evolution_dispatcher.aggregate_and_apply(
                context=self.state.context,
                meta=self.state.final_meta or {},
            )
        except Exception as e:
            self.logger.error(f"Feedback pipeline failed: {e}")

    @staticmethod
    def create_with_all_agents(
        llm_model,
        rag_instance=None,
        permission_manager=None,
        max_iterations: int = int(os.getenv("MAX_ITERATIONS", 100)),
        context_storage=None,
        session_id: str = None,
    ):
        from .qa_agent import QAAgent
        from .planning_agent import PlanningAgent
        from .router_agent import RouterAgent
        from .verification_agent import VerificationAgent
        from .risk_agent import RiskAgent, RiskLevel
        from .task_agents.llm_agent import LLMTaskAgent
        from .task_agents.rag_agent import RAGTaskAgent
        from .task_agents.mcp_agent import MCPTaskAgent

        if rag_instance is None:
            from rag import LocalRAG

            data_path = DATA_PATH
            rag_instance = LocalRAG(llm_model, data_path=data_path)
        if permission_manager is None:
            from permission_manager import PermissionManager, SafetyLevel

            permission_manager = PermissionManager(
                human_approval_threshold=SafetyLevel.HIGH
            )
        from services.feedback_judge import FeedbackJudge
        from services.evolution_dispatcher import EvolutionDispatcher

        memory_repo = getattr(rag_instance, "mongo_repo", None)
        feedback_judge = FeedbackJudge()
        evolution_dispatcher = EvolutionDispatcher(memory_repo=memory_repo)
        runtime = AgentRuntime(
            llm_model,
            max_iterations=max_iterations,
            context_storage=context_storage,
            session_id=session_id,
            feedback_judge=feedback_judge,
            evolution_dispatcher=evolution_dispatcher,
        )

        # Register core agents
        runtime.register_agent("qa", QAAgent(llm_model))
        runtime.register_agent("planning", PlanningAgent(llm_model))
        runtime.register_agent("router", RouterAgent(llm_model))
        runtime.register_agent("verification", VerificationAgent(llm_model))
        runtime.register_agent(
            "risk", RiskAgent(llm_model, risk_threshold=RiskLevel.HIGH)
        )

        # Register task agents
        runtime.register_agent("task_llm", LLMTaskAgent(llm_model))
        runtime.register_agent(
            "task_rag", RAGTaskAgent(llm_model, rag_instance=rag_instance)
        )

        mcp_agent = MCPTaskAgent(llm_model)
        mcp_agent.permission_manager = permission_manager
        runtime.register_agent("task_mcp", mcp_agent)

        from skills import init_skills, registry

        init_skills()
        runtime.state.context["available_skills"] = [
            s.to_dict() for s in registry.list_skills()
        ]

        try:
            from utils.mcp_loader import load_from_env as _load_mcp

            _load_mcp(mcp_agent)
        except Exception:
            pass
        runtime.state.context["available_mcp_tools"] = mcp_agent.get_available_tools()

        from schemas.capability import Capability, CapabilityKind

        caps = []
        for s in registry.list_skills():
            caps.append(
                Capability(
                    id=f"skill.{s.name}",
                    kind=CapabilityKind.SKILL,
                    name=s.name,
                    description=s.description,
                    safety_level="SAFE",
                    permission=None,
                    when_to_use=None,
                    requires_approval=False,
                ).to_dict()
            )
        for t_name, t in mcp_agent.tools.items():
            perm = t.get("permission")
            safety = "UNKNOWN"
            requires = False
            if permission_manager and perm:
                r = permission_manager.check_permission(perm)
                safety = r.get("safety_level", "UNKNOWN")
                requires = bool(r.get("needs_human", False))
            caps.append(
                Capability(
                    id=f"mcp.{t_name}",
                    kind=CapabilityKind.MCP,
                    name=t_name,
                    description=t_name,
                    safety_level=safety,
                    permission=perm,
                    when_to_use=None,
                    requires_approval=requires,
                ).to_dict()
            )
        runtime.state.context["capabilities"] = caps
        hints = []
        for c in caps:
            hints.append(
                {
                    "id": c["id"],
                    "description": c["description"],
                    "when_to_use": c.get("when_to_use"),
                    "safety": c.get("safety_level"),
                    "requires_approval": c.get("requires_approval", False),
                }
            )
        runtime.state.context["planning_hints"] = {"capabilities": hints}
        runtime.register_agent("error_handler", ErrorHandlerAgent(llm_model))

        return runtime
