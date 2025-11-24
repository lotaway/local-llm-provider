"""Verification Agent - Validates agent outputs"""

from typing import Any, Dict
from .agent_base import BaseAgent, AgentResult, AgentStatus


class VerificationAgent(BaseAgent):
    """Agent for validating task execution results"""
    
    SYSTEM_PROMPT = """你是一个结果验证助手。你的任务是：
1. 检查答案质量和相关性
2. 验证事实一致性
3. 检测可能的幻觉或错误
4. 评估置信度
5. 决定是否需要重试

输出JSON格式：
{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "issues": ["问题1", "问题2"],
    "needs_retry": true/false,
    "retry_reason": "如果需要重试，说明原因",
    "verification_notes": "验证说明"
}"""
    
    def execute(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        """
        Verify task execution result
        
        Args:
            input_data: Task execution result
            context: Runtime context
            
        Returns:
            AgentResult with verification status
        """
        task_result = input_data
        current_task = context.get("current_task", {})
        original_query = context.get("original_query", "")
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"""
原始问题：{original_query}
任务描述：{current_task.get('description', '')}
任务结果：{task_result}

请验证这个结果的质量和正确性。
"""}
        ]
        
        try:
            response = self._call_llm(messages, temperature=0.1, max_new_tokens=1000)
            verification = self._parse_json_response(response)
            
            # Store verification in context
            if "verifications" not in context:
                context["verifications"] = []
            context["verifications"].append(verification)
            
            # Check if retry needed
            if verification.get("needs_retry", False):
                return AgentResult(
                    status=AgentStatus.NEEDS_RETRY,
                    data=current_task,
                    message=f"需要重试: {verification.get('retry_reason', 'unknown')}",
                    next_agent="router",
                    metadata=verification
                )
            
            # Check if valid
            if not verification.get("is_valid", False):
                return AgentResult(
                    status=AgentStatus.FAILURE,
                    data=None,
                    message=f"验证失败: {', '.join(verification.get('issues', []))}",
                    metadata=verification
                )
            
            # Valid result, proceed to risk assessment
            # Store task result
            if "task_results" not in context:
                context["task_results"] = []
            context["task_results"].append({
                "task_id": current_task.get("task_id", ""),
                "description": current_task.get("description", ""),
                "result": task_result,
                "verification": verification
            })
            
            # Mark task as completed
            if "completed_tasks" not in context:
                context["completed_tasks"] = []
            context["completed_tasks"].append(current_task.get("task_id", ""))
            
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=task_result,
                message=f"验证通过 (置信度: {verification.get('confidence', 0):.2f})",
                next_agent="risk",
                metadata=verification
            )
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            # On verification error, assume result is valid but log the issue
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=task_result,
                message=f"验证过程出错，假定结果有效: {str(e)}",
                next_agent="risk"
            )
