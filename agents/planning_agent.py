"""Planning Agent - Decomposes tasks and creates execution plans"""

from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus


class PlanningAgent(BaseAgent):
    """Agent for task decomposition and execution planning"""
    
    SYSTEM_PROMPT = """你是一个任务规划助手。你的任务是：
1. 将复杂查询分解为可执行的子任务
2. 确定每个子任务需要的工具/Agent（LLM、RAG、MCP）
3. 创建执行计划，包含任务依赖关系
4. 检查任务完成状态，决定是否继续或结束

可用的任务类型：
- llm: 直接使用LLM回答，适合常识问答、创作、推理
- rag: 使用RAG检索文档回答，适合需要特定知识库的问题
- mcp: 调用外部工具，如搜索、文件操作、图像识别等

输出JSON格式：
{
    "plan": [
        {
            "task_id": "task_1",
            "description": "任务描述",
            "agent_type": "llm|rag|mcp",
            "tool_name": "如果是mcp，指定工具名",
            "dependencies": ["依赖的task_id"],
            "priority": 1
        }
    ],
    "reasoning": "规划理由",
    "estimated_steps": 3
}

如果任务已完成，输出：
{
    "completed": true,
    "final_answer": "最终答案",
    "reasoning": "完成理由"
}"""
    
    def execute(self, input_data: Any, context: Dict[str, Any]) -> AgentResult:
        """
        Create or update execution plan
        
        Args:
            input_data: Parsed query or previous task results
            context: Runtime context with history
            
        Returns:
            AgentResult with execution plan or completion status
        """
        # Get execution history summary
        history_summary = context.get("history_summary", "")
        parsed_query = context.get("parsed_query", {})
        
        # Check if we have task results to evaluate
        task_results = context.get("task_results", [])
        
        if task_results:
            # Evaluate if task is complete
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""
原始问题：{context.get('original_query', '')}
已完成的任务结果：
{self._format_task_results(task_results)}

请判断是否已经完成用户的问题，如果完成则输出final_answer，否则规划下一步任务。
"""}
            ]
        else:
            # Initial planning
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""
用户问题：{context.get('original_query', '')}
问题意图：{parsed_query.get('intent', '')}
问题类型：{parsed_query.get('query_type', '')}

请为这个问题创建执行计划。
"""}
            ]
        
        try:
            response = self._call_llm(messages, temperature=0.2, max_new_tokens=2000)
            plan_data = self._parse_json_response(response)
            
            # Check if completed
            if plan_data.get("completed", False):
                return AgentResult(
                    status=AgentStatus.COMPLETE,
                    data=plan_data.get("final_answer", ""),
                    message="任务已完成",
                    metadata={"reasoning": plan_data.get("reasoning", "")}
                )
            
            # Store plan in context
            context["current_plan"] = plan_data.get("plan", [])
            context["plan_reasoning"] = plan_data.get("reasoning", "")
            
            # Get first task to execute
            plan = plan_data.get("plan", [])
            if not plan:
                return AgentResult(
                    status=AgentStatus.FAILURE,
                    data=None,
                    message="规划失败：没有生成任务"
                )
            
            # Find first task with no dependencies or satisfied dependencies
            next_task = self._get_next_task(plan, context.get("completed_tasks", []))
            
            if not next_task:
                return AgentResult(
                    status=AgentStatus.FAILURE,
                    data=None,
                    message="规划失败：无法找到可执行的任务"
                )
            
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=next_task,
                message=f"规划完成，下一步: {next_task['description']}",
                next_agent="router",
                metadata={"plan": plan}
            )
            
        except Exception as e:
            self.logger.error(f"Planning failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message=f"规划失败: {str(e)}"
            )
    
    def _format_task_results(self, results: List[Dict]) -> str:
        """Format task results for LLM"""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(f"{i}. {result.get('description', 'Unknown')}: {result.get('result', 'No result')}")
        return "\n".join(formatted)
    
    def _get_next_task(self, plan: List[Dict], completed_tasks: List[str]) -> Dict:
        """Get next executable task from plan"""
        for task in plan:
            task_id = task.get("task_id", "")
            if task_id in completed_tasks:
                continue
            
            # Check dependencies
            dependencies = task.get("dependencies", [])
            if all(dep in completed_tasks for dep in dependencies):
                return task
        
        return None
