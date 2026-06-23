from typing import Any, Dict, List, Optional
from .agent_base import BaseAgent, AgentResult, AgentStatus

class PlanningAgent(BaseAgent):
    SYSTEM_PROMPT = """You are a Task Planning Assistant. Your goals:
1. Decompose complex queries into sub-tasks.
2. Determine Agent type (llm|rag|mcp).
3. Create an execution plan with dependencies.
4. **IMPORTANT**: Prioritize using capabilities from "Available Skills" and "Available Tools" lists. Only suggest using 'skill-creator' if existing capabilities cannot cover the requirements.
5. If existing skills can complete the task through combination or multiple calls, do not create redundant new skills.

Output JSON format:
{
    "plan": [{"task_id": "T1", "description": "...", "agent_type": "...", "tool_name": "...", "dependencies": []}],
    "reasoning": "Rationale for selecting these steps and tools",
    "completed": false,
    "final_answer": "Final answer if the task is finished"
}"""

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        messages = self._prepare_messages(input_data, context)
        
        try:
            llm_res = await self._call_llm(messages, stream_callback, temperature=0.2)
            return self._handle_response(llm_res, context)
        except Exception as e:
            return AgentResult(AgentStatus.FAILURE, None, f"PlanningError: {str(e)}")

    def _prepare_messages(self, input_data: Any, context: Dict[str, Any]) -> List[Dict]:
        info = self._get_env_info(context)
        content = f"Query: {context.get('original_query')}\n{info}"
        
        if isinstance(input_data, dict) and input_data.get("error") == "tool_not_found":
            content += f"\nRetry Reason: {input_data.get('suggestion')}"

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

    def _get_env_info(self, context: Dict[str, Any]) -> str:
        from skills import registry
        skills = [f"- {s.name}: {s.description}" for s in registry.list_skills()]
        mcp = context.get("available_mcp_tools", [])
        
        return f"Available Skills:\n{chr(10).join(skills)}\nAvailable Tools: {', '.join(mcp)}"

    def _handle_response(self, llm_res: Dict[str, str], context: Dict[str, Any]) -> AgentResult:
        plan_data = self._parse_json_response(llm_res["response"])
        thought = llm_res["thought"]
        
        if plan_data.get("completed"):
            return AgentResult(AgentStatus.COMPLETE, plan_data.get("final_answer"), thought_process=thought)
            
        plan = plan_data.get("plan", [])
        context["current_plan"] = plan
        
        next_task = self._segment_next_task(plan, context.get("completed_tasks", []))
        if not next_task:
            return AgentResult(AgentStatus.FAILURE, None, "NoExecutableTaskFound")
            
        return AgentResult(AgentStatus.SUCCESS, next_task, next_agent="router", thought_process=thought)

    def _segment_next_task(self, plan: List[Dict], completed: List[str]) -> Optional[Dict]:
        for task in plan:
            tid = task.get("task_id")
            if tid in completed:
                continue
            if all(dep in completed for dep in task.get("dependencies", [])):
                return task
        return None
