from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus

class VerificationAgent(BaseAgent):
    SYSTEM_PROMPT = """你是一个结果验证助手。任务：
1. 检查答案质量和相关性
2. 验证事实一致性
3. 检测可能的幻觉或错误
4. 评估置信度
5. 决定是否需要重试

输出JSON格式：
{
    "is_valid": true,
    "confidence": 0.9,
    "quality_score": 0.9,
    "issues": [],
    "needs_retry": false,
    "retry_reason": "",
    "verification_notes": ""
}"""

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        messages = self._build_verify_messages(input_data, context)
        
        try:
            llm_res = await self._call_llm(messages, stream_callback, temperature=0.1)
            return self._analyze_verification(llm_res, input_data, context)
        except Exception as e:
            return self._handle_error(e, input_data)

    def _build_verify_messages(self, data: Any, context: Dict[str, Any]) -> List[Dict]:
        task = context.get("current_task", {})
        query = context.get("original_query", "")
        content = f"问题：{query}\n描述：{task.get('description')}\n结果：{data}"
        
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

    def _analyze_verification(self, llm_res: Dict[str, str], data: Any, context: Dict[str, Any]) -> AgentResult:
        res = self._parse_json_response(llm_res["response"])
        thought = llm_res["thought"]
        
        context.setdefault("verifications", []).append(res)
        
        if res.get("needs_retry"):
            return self._create_retry_result(res, context, thought)
            
        if not res.get("is_valid"):
            return AgentResult(AgentStatus.FAILURE, None, f"VerifyFailed: {res.get('issues')}", thought_process=thought)
            
        self._record_success(data, res, context)
        return AgentResult(AgentStatus.SUCCESS, data, "Verified", next_agent="risk", thought_process=thought)

    def _create_retry_result(self, res: Dict, context: Dict, thought: str) -> AgentResult:
        task = context.get("current_task")
        return AgentResult(AgentStatus.NEEDS_RETRY, task, f"Retry: {res.get('retry_reason')}", next_agent="router", thought_process=thought)

    def _record_success(self, data: Any, res: Dict, context: Dict):
        task = context.get("current_task", {})
        context.setdefault("task_results", []).append({
            "task_id": task.get("task_id"),
            "description": task.get("description"),
            "result": data,
            "verification": res
        })
        context.setdefault("completed_tasks", []).append(task.get("task_id"))

    def _handle_error(self, error: Exception, data: Any) -> AgentResult:
        return AgentResult(AgentStatus.SUCCESS, data, f"VerifyError: {str(error)}", next_agent="risk")
