from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus


class PlanningAgent(BaseAgent):
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
- 优先选择已存在的技能（Skills）和工具（Tools）。
- 如果没有可用的MCP工具或技能来满足用户需求，你可以建议使用 'skill-creator' 技能来创建一个新的技能。
- 'skill-creator' 是一个特殊的技能，它可以帮助你定义、编写、测试和打包新的技能到技能库中。
- 当你需要实现一个复杂、重复、且当前系统无法直接处理的流程时，考虑创建一个新技能。
- 不要建议使用不存在的工具，除非你打算使用 'skill-creator' 来创建它。

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

    def _get_available_skills_info(self) -> str:
        try:
            from skills import registry

            skills_info = []
            tools_info = []

            for skill in registry.list_skills():
                skills_info.append(f"- {skill.name}: {skill.description}")
                for tool in skill.tools:
                    tools_info.append(f"  - {tool.name}: {tool.description}")

            skills_str = "\n".join(skills_info) if skills_info else "无"
            tools_str = "\n".join(tools_info) if tools_info else "无"

            return f"""
可用的技能（Skills）:
{skills_str}

每个技能提供的工具（Tools）:
{tools_str}
"""
        except Exception:
            return "技能信息无法获取"

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        parsed_query = context.get("parsed_query", {})
        available_mcp_tools = context.get("available_mcp_tools", [])

        if "file_map" in context and "read_file" not in available_mcp_tools:
            available_mcp_tools.append("read_file")

        mcp_tools_info = (
            f"可用的MCP工具: {', '.join(available_mcp_tools)}"
            if available_mcp_tools
            else "当前没有可用的MCP工具，请只使用LLM和RAG能力"
        )

        skills_info = self._get_available_skills_info()
        available_files = context.get("available_files", [])
        files_info = ""
        if available_files:
            files_info = (
                "\n可用的文件 (使用 'read_file' 工具读取内容):\n"
                + "\n".join(available_files)
                + "\n"
            )

        self.logger.info(f"Planning agent started")
        is_mcp_retry = (
            isinstance(input_data, dict) and input_data.get("error") == "tool_not_found"
        )

        task_results = context.get("task_results", [])

        if is_mcp_retry:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
原始问题：{context.get('original_query', '')}
之前尝试的任务：{input_data.get('original_task', '')}
失败原因：{input_data.get('suggestion', '')}

{mcp_tools_info}
{skills_info}
{files_info}

请重新规划任务，只使用可用的工具（LLM、RAG或已注册的MCP工具）。
""",
                },
            ]
        elif task_results:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
原始问题：{context.get('original_query', '')}
已完成的任务结果：
{self._format_task_results(task_results)}

{mcp_tools_info}
{skills_info}
{files_info}

请判断是否已经完成用户的问题，如果完成则输出final_answer，否则规划下一步任务。
""",
                },
            ]
        else:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"""
用户问题：{context.get('original_query', '')}
问题意图：{parsed_query.get('intent', '')}
问题类型：{parsed_query.get('query_type', '')}

{mcp_tools_info}
{skills_info}
{files_info}

请为这个问题创建执行计划。
""",
                },
            ]

        try:
            response = await self._call_llm(
                messages,
                stream_callback=stream_callback,
                temperature=0.2,
                max_new_tokens=2000,
            )

            plan_data = self._parse_json_response(response)

            if plan_data.get("completed", False):
                return AgentResult(
                    status=AgentStatus.COMPLETE,
                    data=plan_data.get("final_answer", ""),
                    message="任务已完成",
                    metadata={"reasoning": plan_data.get("reasoning", "")},
                )

            context["current_plan"] = plan_data.get("plan", [])
            context["plan_reasoning"] = plan_data.get("reasoning", "")

            plan = plan_data.get("plan", [])
            if not plan:
                fallback_task = {
                    "task_id": "fallback_llm_1",
                    "description": f"使用LLM直接回答: {context.get('original_query', input_data)}",
                    "agent_type": "llm",
                    "dependencies": [],
                    "priority": 1,
                }
                return AgentResult(
                    status=AgentStatus.SUCCESS,
                    data=fallback_task,
                    message=f"使用备用方案: LLM直接回答",
                    next_agent="router",
                    metadata={"plan": [fallback_task], "fallback": True},
                )

            next_task = self._get_next_task(plan, context.get("completed_tasks", []))

            if not next_task:
                completed_tasks = context.get("completed_tasks", [])
                if not completed_tasks:
                    fallback_task = {
                        "task_id": "fallback_llm_1",
                        "description": f"使用LLM直接回答: {context.get('original_query', input_data)}",
                        "agent_type": "llm",
                        "dependencies": [],
                        "priority": 1,
                    }
                    return AgentResult(
                        status=AgentStatus.SUCCESS,
                        data=fallback_task,
                        message=f"使用备用方案: LLM直接回答",
                        next_agent="router",
                        metadata={"plan": [fallback_task], "fallback": True},
                    )

                return AgentResult(
                    status=AgentStatus.FAILURE,
                    data=None,
                    message="规划失败：无法找到可执行的任务",
                )

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=next_task,
                message=f"规划完成，下一步: {next_task['description']}",
                next_agent="router",
                metadata={"plan": plan},
            )

        except Exception as e:
            return AgentResult(
                status=AgentStatus.FAILURE, data=None, message=f"规划失败: {str(e)}"
            )

    def _format_task_results(self, results: List[Dict]) -> str:
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"{i}. {result.get('description', 'Unknown')}: {result.get('result', 'No result')}"
            )
        return "\n".join(formatted)

    def _get_next_task(self, plan: List[Dict], completed_tasks: List[str]) -> Dict:
        for task in plan:
            task_id = task.get("task_id", "")
            if task_id in completed_tasks:
                continue

            dependencies = task.get("dependencies", [])
            if all(dep in completed_tasks for dep in dependencies):
                return task

        return None
