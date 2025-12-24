"""Risk Agent - Assesses operation risk and human intervention needs"""

from typing import Any, Dict
from .agent_base import BaseAgent, AgentResult, AgentStatus


class RiskLevel:
    """Risk level constants"""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAgent(BaseAgent):
    """Agent for assessing operation risk and determining human intervention needs"""

    SYSTEM_PROMPT = """你是一个风险评估助手。你的任务是：
1. 评估操作的安全级别
2. 检查是否需要特殊权限
3. 判断是否需要人工审批
4. 识别潜在风险

风险级别定义：
- safe: 完全安全，如只读查询
- low: 低风险，如简单计算
- medium: 中等风险，如文件读取
- high: 高风险，如文件修改、网络请求
- critical: 关键风险，如系统命令、删除操作

输出JSON格式：
{
    "risk_level": "safe|low|medium|high|critical",
    "needs_human": true/false,
    "risk_factors": ["风险因素1", "风险因素2"],
    "required_permissions": ["权限1", "权限2"],
    "recommendation": "建议",
    "human_prompt": "如果需要人工介入，给人类的提示信息"
}"""

    def __init__(
        self, llm_model, name: str = None, risk_threshold: str = RiskLevel.HIGH
    ):
        """
        Initialize risk agent

        Args:
            llm_model: LocalLLModel instance
            name: Agent name
            risk_threshold: Risk level that triggers human intervention
        """
        super().__init__(llm_model, name)
        self.risk_threshold = risk_threshold

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        """
        Assess operation risk

        Args:
            input_data: Task result from verification
            context: Runtime context
            stream_callback: Optional callback for streaming LLM outputs

        Returns:
            AgentResult with risk assessment
        """
        current_task = context.get("current_task", {})
        task_result = input_data

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
任务类型：{current_task.get('agent_type', '')}
任务描述：{current_task.get('description', '')}
工具名称：{current_task.get('tool_name', 'N/A')}
任务结果：{task_result}

请评估这个操作的风险级别。
""",
            },
        ]

        try:
            response = await self._call_llm(
                messages,
                stream_callback=stream_callback,
                temperature=0.1,
                max_new_tokens=1000,
            )
            risk_assessment = self._parse_json_response(response)

            risk_level = risk_assessment.get("risk_level", RiskLevel.MEDIUM)
            needs_human = risk_assessment.get("needs_human", False)

            # Override based on threshold
            if self._should_require_human(risk_level):
                needs_human = True

            # Store risk assessment
            context["last_risk_assessment"] = risk_assessment

            if needs_human:
                return AgentResult(
                    status=AgentStatus.NEEDS_HUMAN,
                    data={
                        "task": current_task,
                        "result": task_result,
                        "risk_assessment": risk_assessment,
                        "prompt": risk_assessment.get(
                            "human_prompt", "需要人工审批此操作"
                        ),
                    },
                    message=f"需要人工介入 (风险级别: {risk_level})",
                    next_agent="planning",
                    metadata=risk_assessment,
                )

            # Safe to proceed, go back to planning for next task
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=task_result,
                message=f"风险评估通过 (级别: {risk_level})",
                next_agent="planning",
                metadata=risk_assessment,
            )

        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            # On error, be conservative and require human intervention
            return AgentResult(
                status=AgentStatus.NEEDS_HUMAN,
                data={
                    "task": current_task,
                    "result": task_result,
                    "error": str(e),
                    "prompt": f"风险评估失败，需要人工审核: {str(e)}",
                },
                message=f"风险评估失败: {str(e)}",
                next_agent="planning",
            )

    def _should_require_human(self, risk_level: str) -> bool:
        """Determine if risk level requires human intervention"""
        risk_order = [
            RiskLevel.SAFE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]

        try:
            current_index = risk_order.index(risk_level)
            threshold_index = risk_order.index(self.risk_threshold)
            return current_index >= threshold_index
        except ValueError:
            # Unknown risk level, be conservative
            return True
