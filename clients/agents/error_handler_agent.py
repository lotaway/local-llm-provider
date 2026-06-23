import json
from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus
from .error_utils import EvidenceExtractor, SystemSnapshot

class ErrorHandlerAgent(BaseAgent):
    ROOT_CAUSE_TAXONOMY = {
        "LLM_TIMEOUT": "LLM service unresponsive",
        "LLM_PARSE_ERROR": "Failed to parse LLM response",
        "TOOL_EXECUTION_ERROR": "External tool failed",
        "PERMISSION_DENIED": "Insufficient permissions",
        "INVALID_PLAN": "Impossible plan",
        "UNKNOWN_ERROR": "Uncategorized error"
    }

    SYSTEM_PROMPT = """你是一个异常归因专家。任务：
1. 分析证据包和系统快照
2. 匹配根因分类库 (Taxonomy)
3. 评估置信度和动作风险

Taxonomy: {taxonomy}

输出JSON:
{{
    "root_cause": "KEY",
    "confidence": 0.9,
    "explanation": "...",
    "action": {{"desc": "...", "risk": 0.1}},
    "decision": "AUTOMATIC | MANUAL"
}}"""

    def __init__(self, llm_model, name: str = None):
        super().__init__(llm_model, name)
        self.tax_json = json.dumps(self.ROOT_CAUSE_TAXONOMY, ensure_ascii=False)

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        evidence = EvidenceExtractor.extract_from_exception(input_data.get("exception"), input_data.get("agent_name"), context)
        snapshot = SystemSnapshot.capture(context)
        
        messages = self._build_messages(evidence, snapshot)
        
        try:
            llm_res = await self._call_llm(messages, stream_callback, temperature=0.0)
            return self._decide_action(llm_res, context)
        except Exception as e:
            return AgentResult(AgentStatus.NEEDS_HUMAN, {"error": str(e)}, "HandlerFailed", next_agent="planning")

    def _build_messages(self, evidence: Dict, snapshot: Dict) -> List[Dict]:
        content = f"证据: {json.dumps(evidence)}\n快照: {json.dumps(snapshot)}"
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(taxonomy=self.tax_json)},
            {"role": "user", "content": content}
        ]

    def _decide_action(self, llm_res: Dict[str, str], context: Dict[str, Any]) -> AgentResult:
        analysis = self._parse_json_response(llm_res["response"])
        thought = llm_res["thought"]
        
        conf = analysis.get("confidence", 0.0)
        risk = analysis.get("action", {}).get("risk", 1.0)
        
        if self._is_auto_permitted(conf, risk, analysis.get("decision")):
            return self._create_auto_result(analysis, context, thought)
        
        return AgentResult(AgentStatus.NEEDS_HUMAN, analysis, "ManualInterventionRequired", next_agent="planning", thought_process=thought)

    def _is_auto_permitted(self, confidence: float, risk: float, decision: str) -> bool:
        if decision != "AUTOMATIC":
            return False
        if confidence >= 0.95 and risk <= 0.2:
            return True
        return confidence >= 0.9 and risk <= 0.1

    def _create_auto_result(self, analysis: Dict, context: Dict, thought: str) -> AgentResult:
        context["last_error_fix"] = analysis.get("action")
        return AgentResult(
            AgentStatus.CONTINUE, 
            analysis.get("action"), 
            "AutoResolved", 
            next_agent="planning", 
            thought_process=thought
        )
