from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus

class RiskLevel:
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskAgent(BaseAgent):
    SYSTEM_PROMPT = """你是一个风险评估专家。任务：
1. 评估安全级别 (safe|low|medium|high|critical)
2. 检查权限和人工审批需求
3. 识别风险因素

输出JSON:
{
    "risk_level": "...",
    "needs_human": false,
    "factors": [],
    "permissions": [],
    "human_prompt": "..."
}"""

    def __init__(self, llm_model, name: str = None, risk_threshold: str = RiskLevel.HIGH):
        super().__init__(llm_model, name)
        self.risk_threshold = risk_threshold
        self.risk_order = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        messages = self._build_risk_messages(input_data, context)
        
        try:
            llm_res = await self._call_llm(messages, stream_callback, temperature=0.1)
            return self._finalize_assessment(llm_res, input_data, context)
        except Exception as e:
            return self._handle_risk_error(e, input_data, context)

    def _build_risk_messages(self, data: Any, context: Dict[str, Any]) -> List[Dict]:
        task = context.get("current_task", {})
        content = f"类型：{task.get('agent_type')}\n描述：{task.get('description')}\n结果：{data}"
        
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

    def _finalize_assessment(self, llm_res: Dict[str, str], data: Any, context: Dict[str, Any]) -> AgentResult:
        res = self._parse_json_response(llm_res["response"])
        thought = llm_res["thought"]
        
        level = res.get("risk_level", RiskLevel.MEDIUM)
        should_human = res.get("needs_human") or self._exceeds_threshold(level)
        
        context["last_risk_assessment"] = res
        
        if should_human:
            return self._create_human_result(res, data, context, level, thought)
            
        return AgentResult(AgentStatus.SUCCESS, data, f"RiskPass: {level}", next_agent="planning", thought_process=thought)

    def _exceeds_threshold(self, level: str) -> bool:
        try:
            return self.risk_order.index(level) >= self.risk_order.index(self.risk_threshold)
        except ValueError:
            return True

    def _create_human_result(self, res, data, context, level, thought) -> AgentResult:
        return AgentResult(
            AgentStatus.NEEDS_HUMAN,
            {"task": context.get("current_task"), "result": data, "assessment": res},
            message=f"RiskTier: {level}",
            next_agent="planning",
            thought_process=thought
        )

    def _handle_risk_error(self, error: Exception, data: Any, context: Dict[str, Any]) -> AgentResult:
        return AgentResult(
            AgentStatus.NEEDS_HUMAN,
            {"error": str(error), "task": context.get("current_task")},
            message="RiskAssessmentFailed",
            next_agent="planning"
        )
