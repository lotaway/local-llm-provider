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

重要提示：
- 只能使用已注册的MCP工具，可用的MCP工具列表会在用户消息中提供
- 如果没有可用的MCP工具，请只使用LLM和RAG能力来完成任务
- 不要建议使用不存在的MCP工具

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
    
    def execute(self, input_data: Any, context: Dict[str, Any], stream_callback=None) -> AgentResult:
        """
        Create or update execution plan
        
        Args:
            input_data: Parsed query or previous task results
            context: Runtime context with history
            stream_callback: Optional callback for streaming LLM outputs
            
        Returns:
            AgentResult with execution plan or completion status
        """
        # Get execution history summary
        history_summary = context.get("history_summary", "")
        parsed_query = context.get("parsed_query", {})
        
        # Get available MCP tools
        available_mcp_tools = context.get("available_mcp_tools", [])
        # Add read_file if files are available
        if "file_map" in context and "read_file" not in available_mcp_tools:
             available_mcp_tools.append("read_file")
             
        mcp_tools_info = f"可用的MCP工具: {', '.join(available_mcp_tools)}" if available_mcp_tools else "当前没有可用的MCP工具，请只使用LLM和RAG能力"
        
        # Add available files info
        available_files = context.get("available_files", [])
        files_info = ""
        if available_files:
            files_info = "\n可用的文件 (使用 'read_file' 工具读取内容):\n" + "\n".join(available_files) + "\n"
        
        self.logger.info(f"Planning agent started")
        self.logger.debug(f"  Available MCP tools: {available_mcp_tools}")
        self.logger.debug(f"  Context keys: {list(context.keys())}")
        
        # Check if this is a retry due to MCP tool unavailable
        is_mcp_retry = isinstance(input_data, dict) and input_data.get("error") == "tool_not_found"
        if is_mcp_retry:
            self.logger.warning(f"Replanning due to MCP tool unavailable: {input_data.get('original_task', '')}")
        
        # Check if we have task results to evaluate
        task_results = context.get("task_results", [])
        if task_results:
            self.logger.info(f"Evaluating {len(task_results)} completed task results")
        
        if is_mcp_retry:
            # Replanning due to MCP tool unavailable
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""
原始问题：{context.get('original_query', '')}
之前尝试的任务：{input_data.get('original_task', '')}
失败原因：{input_data.get('suggestion', '')}

{mcp_tools_info}
{files_info}

请重新规划任务，只使用可用的工具（LLM、RAG或已注册的MCP工具）。
"""}
            ]
        elif task_results:
            # Evaluate if task is complete
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"""
原始问题：{context.get('original_query', '')}
已完成的任务结果：
{self._format_task_results(task_results)}

{mcp_tools_info}
{files_info}

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

{mcp_tools_info}
{files_info}

请为这个问题创建执行计划。
"""}
            ]
        
        try:
            self.logger.debug("Calling LLM for planning...")
            response = self._call_llm(messages, stream_callback=stream_callback, temperature=0.2, max_new_tokens=2000)
            
            self.logger.debug(f"LLM response received (length: {len(response)} chars)")
            self.logger.debug(f"Raw LLM response: {response[:500]}..." if len(response) > 500 else f"Raw LLM response: {response}")
            
            plan_data = self._parse_json_response(response)
            self.logger.debug(f"Parsed plan data: {plan_data}")
            
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
            self.logger.info(f"Generated plan with {len(plan)} tasks")
            
            if not plan:
                self.logger.error("Planning failed: LLM returned empty plan")
                self.logger.error(f"  Plan data: {plan_data}")
                self.logger.error(f"  Context: original_query={context.get('original_query', 'N/A')}")
                
                # Fallback: create a simple LLM task
                self.logger.warning("Falling back to simple LLM task")
                fallback_task = {
                    "task_id": "fallback_llm_1",
                    "description": f"使用LLM直接回答: {context.get('original_query', input_data)}",
                    "agent_type": "llm",
                    "dependencies": [],
                    "priority": 1
                }
                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    data=fallback_task,
                    message=f"使用备用方案: LLM直接回答",
                    next_agent="router",
                    metadata={"plan": [fallback_task], "fallback": True}
                )
            
            # Find first task with no dependencies or satisfied dependencies
            next_task = self._get_next_task(plan, context.get("completed_tasks", []))
            
            if not next_task:
                completed_tasks = context.get("completed_tasks", [])
                self.logger.error("Planning failed: No executable task found")
                self.logger.error(f"  Total tasks in plan: {len(plan)}")
                self.logger.error(f"  Completed tasks: {completed_tasks}")
                self.logger.error(f"  Plan details:")
                for i, task in enumerate(plan):
                    self.logger.error(f"    Task {i+1}: {task.get('task_id')} - {task.get('description')}")
                    self.logger.error(f"      Dependencies: {task.get('dependencies', [])}")
                    self.logger.error(f"      Agent type: {task.get('agent_type')}")
                
                # Fallback: create a simple LLM task if no tasks were completed yet
                if not completed_tasks:
                    self.logger.warning("No tasks completed yet, falling back to simple LLM task")
                    fallback_task = {
                        "task_id": "fallback_llm_1",
                        "description": f"使用LLM直接回答: {context.get('original_query', input_data)}",
                        "agent_type": "llm",
                        "dependencies": [],
                        "priority": 1
                    }
                    return AgentResult(
                        status=AgentStatus.SUCCESS,
                        data=fallback_task,
                        message=f"使用备用方案: LLM直接回答",
                        next_agent="router",
                        metadata={"plan": [fallback_task], "fallback": True}
                    )
                
                return AgentResult(
                    status=AgentStatus.FAILURE,
                    data=None,
                    message="规划失败：无法找到可执行的任务"
                )
            
            self.logger.info(f"Next task selected: {next_task.get('task_id')} - {next_task.get('description')}")
            self.logger.debug(f"  Task details: {next_task}")
            
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=next_task,
                message=f"规划完成，下一步: {next_task['description']}",
                next_agent="router",
                metadata={"plan": plan}
            )
            
        except Exception as e:
            self.logger.error(f"Planning failed with exception: {e}", exc_info=True)
            self.logger.error(f"  Input data: {input_data}")
            self.logger.error(f"  Context keys: {list(context.keys())}")
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
