from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus
from constants import DATA_PATH

class RAGTaskAgent(BaseAgent):
    def __init__(self, llm_model, rag_instance=None, name: str = None):
        super().__init__(llm_model, name)
        self.rag = rag_instance

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        query = self._format_query(input_data, context)
        
        try:
            self._ensure_rag()
            res = await self.rag.generate_answer(query, stream_callback)
            return self._wrap_result(res, context)
        except Exception as e:
            return AgentResult(AgentStatus.FAILURE, None, f"RAGError: {str(e)}")

    def _format_query(self, input_data: Any, context: Dict[str, Any]) -> str:
        task = input_data if isinstance(input_data, dict) else {}
        desc = task.get("description", "")
        origin = context.get("original_query", "")
        return f"{origin}\nTask: {desc}" if desc else origin

    def _ensure_rag(self):
        if self.rag:
            return
        from rag import LocalRAG
        self.rag = LocalRAG(self.llm, data_path=DATA_PATH)

    def _wrap_result(self, res: Dict[str, str], context: Dict[str, Any]) -> AgentResult:
        context["last_agent_type"] = "rag"
        if hasattr(self.rag, "last_retrieved_chunk_ids"):
            context["last_retrieved_chunk_ids"] = self.rag.last_retrieved_chunk_ids or []
            
        context.setdefault("skills_used", []).append("rag")
        
        return AgentResult(
            AgentStatus.SUCCESS, 
            res.get("answer"), 
            "RAGTaskComplete", 
            next_agent="verification", 
            thought_process=res.get("thought")
        )
