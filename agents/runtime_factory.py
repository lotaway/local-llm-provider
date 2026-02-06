import os
from typing import List, Dict, Any, Optional
from agents.agent_runtime import AgentRuntime
from constants import DATA_PATH
from schemas.capability import Capability, CapabilityKind

class RuntimeFactory:
    @staticmethod
    def create_with_all_agents(
        llm_model: Any,
        rag_instance: Any = None,
        permission_manager: Any = None,
        max_iterations: int = 100,
        context_storage: Any = None,
        session_id: Optional[str] = None,
        trace_collector: Any = None
    ) -> AgentRuntime:
        rag = RuntimeFactory._ensure_rag(llm_model, rag_instance)
        perm = RuntimeFactory._ensure_permission_manager(permission_manager)
        
        runtime = RuntimeFactory._build_runtime_instance(
            llm_model, max_iterations, context_storage, session_id, rag
        )
        
        if trace_collector:
            runtime.set_trace_collector(trace_collector)

        RuntimeFactory._register_agents(runtime, llm_model, rag, perm)
        RuntimeFactory._setup_context(runtime, perm)

        return runtime

    @staticmethod
    def _build_runtime_instance(llm, max_iters, storage, session_id, rag) -> AgentRuntime:
        from services.feedback_judge import FeedbackJudge
        from services.evolution_dispatcher import EvolutionDispatcher
        
        memory_repo = getattr(rag, "mongo_repo", None)
        return AgentRuntime(
            llm,
            max_iterations=max_iters,
            context_storage=storage,
            session_id=session_id,
            feedback_judge=FeedbackJudge(),
            evolution_dispatcher=EvolutionDispatcher(memory_repo=memory_repo)
        )

    @staticmethod
    def _ensure_rag(llm, instance):
        if instance is not None:
            return instance
        from rag import LocalRAG
        return LocalRAG(llm, data_path=DATA_PATH)

    @staticmethod
    def _ensure_permission_manager(pm):
        if pm is not None:
            return pm
        from permission_manager import PermissionManager, SafetyLevel
        return PermissionManager(human_approval_threshold=SafetyLevel.HIGH)

    @staticmethod
    def _register_agents(runtime, llm, rag, perm):
        RuntimeFactory._register_core(runtime, llm)
        RuntimeFactory._register_tasks(runtime, llm, rag, perm)

    @staticmethod
    def _register_core(runtime, llm):
        from .qa_agent import QAAgent
        from .planning_agent import PlanningAgent
        from .router_agent import RouterAgent
        from .verification_agent import VerificationAgent
        from .risk_agent import RiskAgent, RiskLevel
        from .error_handler_agent import ErrorHandlerAgent

        runtime.register_agent("qa", QAAgent(llm))
        runtime.register_agent("planning", PlanningAgent(llm))
        runtime.register_agent("router", RouterAgent(llm))
        runtime.register_agent("verification", VerificationAgent(llm))
        runtime.register_agent("risk", RiskAgent(llm, risk_threshold=RiskLevel.HIGH))
        runtime.register_agent("error_handler", ErrorHandlerAgent(llm))

    @staticmethod
    def _register_tasks(runtime, llm, rag, perm):
        from .task_agents.llm_agent import LLMTaskAgent
        from .task_agents.rag_agent import RAGTaskAgent
        from .task_agents.mcp_agent import MCPTaskAgent

        runtime.register_agent("task_llm", LLMTaskAgent(llm))
        runtime.register_agent("task_rag", RAGTaskAgent(llm, rag_instance=rag))
        
        mcp = MCPTaskAgent(llm)
        mcp.permission_manager = perm
        runtime.register_agent("task_mcp", mcp)
        
        RuntimeFactory._load_mcp(mcp, runtime)

    @staticmethod
    def _load_mcp(mcp, runtime):
        try:
            from utils.mcp_loader import load_from_env
            load_from_env(mcp)
        except Exception:
            pass
        runtime.state.context["available_mcp_tools"] = mcp.get_available_tools()

    @staticmethod
    def _setup_context(runtime, perm):
        from skills import init_skills, registry
        init_skills()
        
        skills = registry.list_skills()
        runtime.state.context["available_skills"] = [s.to_dict() for s in skills]
        
        caps = RuntimeFactory._build_capabilities(skills, runtime.agents.get("task_mcp"), perm)
        runtime.state.context["capabilities"] = caps
        runtime.state.context["planning_hints"] = {"capabilities": caps}

    @staticmethod
    def _build_capabilities(skills, mcp_agent, perm) -> List[Dict]:
        caps = []
        for s in skills:
            caps.append(RuntimeFactory._skill_to_cap(s))
        
        if mcp_agent:
            for t_name, t in mcp_agent.tools.items():
                caps.append(RuntimeFactory._mcp_to_cap(t_name, t, perm))
        return caps

    @staticmethod
    def _skill_to_cap(s) -> Dict:
        return Capability(
            id=f"skill.{s.name}",
            kind=CapabilityKind.SKILL,
            name=s.name,
            description=s.description,
            safety_level="SAFE"
        ).to_dict()

    @staticmethod
    def _mcp_to_cap(name, tool, perm) -> Dict:
        p_name = tool.get("permission")
        safety = "UNKNOWN"
        needs_human = False
        if perm and p_name:
            r = perm.check_permission(p_name)
            safety = r.get("safety_level", "UNKNOWN")
            needs_human = bool(r.get("needs_human", False))
            
        return Capability(
            id=f"mcp.{name}",
            kind=CapabilityKind.MCP,
            name=name,
            description=name,
            safety_level=safety,
            permission=p_name,
            requires_approval=needs_human
        ).to_dict()
