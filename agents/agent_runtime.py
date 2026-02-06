import logging
import asyncio
from typing import Dict, Any, List, Optional
from .agent_base import BaseAgent, AgentResult, AgentStatus
from .context_storage import ContextStorage
from .runtime_state import RuntimeState, RuntimeStatus

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
        self.trace_collector = None
        self.human_callback = None

        if self.session_id and self.context_storage:
            self._load_saved_state()

    def set_trace_collector(self, collector):
        self.trace_collector = collector

    def set_human_callback(self, callback):
        self.human_callback = callback

    def register_agent(self, name: str, agent: BaseAgent):
        self.agents[name] = agent

    def get_state(self) -> RuntimeState:
        return self.state

    def reset(self):
        self.state = RuntimeState(max_iterations=self.state.max_iterations)

    async def execute(
        self,
        initial_input: Any,
        start_agent: str = "qa",
        stream_callback=None,
        initial_context: Dict[str, Any] = None,
    ) -> RuntimeState:
        self.state = RuntimeState(max_iterations=self.state.max_iterations)
        self.state.context["initial_input"] = initial_input
        if initial_context:
            self.state.context.update(initial_context)
        self.state.current_agent = start_agent

        if self.trace_collector:
            self.trace_collector.start_workflow(self.session_id or "default", str(initial_input))

        result = await self._run_loop(initial_input, stream_callback)
        self._save_current_state()
        return result

    async def _run_loop(self, current_input: Any, stream_callback=None) -> RuntimeState:
        while self.state.status == RuntimeStatus.RUNNING:
            if self.state.iteration_count >= self.state.max_iterations:
                self._handle_iteration_limit()
                break

            self.state.iteration_count += 1
            agent = self.agents.get(self.state.current_agent)
            
            if not agent:
                self._handle_missing_agent(self.state.current_agent)
                break

            result = await self._step_agent(agent, current_input, stream_callback)
            current_input = await self._process_result(result)
            
        return self.state

    async def _step_agent(self, agent, current_input, stream_callback) -> AgentResult:
        private = self.state.private_contexts.get(agent.name, {})
        result = await agent.execute(
            current_input, self.state.context, private, stream_callback
        )
        
        if result.private_data:
            self.state.private_contexts.setdefault(agent.name, {}).update(result.private_data)
            
        self.state.add_to_history(agent.name, result)
        self._notify_trace_collector(agent.name, current_input, result)
        return result

    def _notify_trace_collector(self, name, input_pay, result):
        if not self.trace_collector:
            return
            
        from schemas.evolution_trace import AgentExecutionTrace, TraceStatus
        self.trace_collector.record_agent_execution(AgentExecutionTrace(
            agent_name=name,
            input_payload=input_pay,
            output_message=result.message,
            status=TraceStatus.SUCCESS if result.status != AgentStatus.FAILURE else TraceStatus.FAILURE,
            thought_process=result.thought_process,
            tool_calls=result.tool_calls
        ))

    async def _process_result(self, result: AgentResult) -> Any:
        handlers = {
            AgentStatus.COMPLETE: self._on_complete,
            AgentStatus.NEEDS_HUMAN: self._on_human_required,
            AgentStatus.FAILURE: self._on_failure,
            AgentStatus.NEEDS_RETRY: self._on_retry,
            AgentStatus.SUCCESS: self._on_continue,
            AgentStatus.CONTINUE: self._on_continue,
        }
        handler = handlers.get(result.status)
        if handler:
            return await handler(result)
        return result.data

    async def _on_complete(self, result: AgentResult) -> Any:
        self.state.status = RuntimeStatus.COMPLETED
        self.state.final_result = result.data
        if self.trace_collector:
            self.trace_collector.persist_workflow(str(result.data))
        return result.data

    async def _on_failure(self, result: AgentResult) -> Any:
        self.state.status = RuntimeStatus.FAILED
        self.state.error_message = result.message
        return result.data

    async def _on_human_required(self, result: AgentResult) -> Any:
        self.state.status = RuntimeStatus.WAITING_HUMAN
        self.state.current_agent = result.next_agent or "planning"
        return result.data

    async def _on_retry(self, result: AgentResult) -> Any:
        return result.data

    async def _on_continue(self, result: AgentResult) -> Any:
        if result.next_agent:
            self.state.current_agent = result.next_agent
        return result.data

    def _handle_iteration_limit(self):
        self.state.status = RuntimeStatus.MAX_ITERATIONS
        self.state.iteration_count_round += 1
        self.state.error_message = "IterationLimitReached"

    def _handle_missing_agent(self, name):
        self.state.status = RuntimeStatus.FAILED
        self.state.error_message = f"AgentNotFound: {name}"

    def _load_saved_state(self):
        saved = self.context_storage.load(self.session_id)
        if saved:
            self.state = RuntimeState.from_dict(saved)

    def _save_current_state(self):
        if self.session_id and self.context_storage:
            self.context_storage.save(self.session_id, self.state.to_dict())
