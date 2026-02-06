from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus

class QAAgent(BaseAgent):
    SYSTEM_PROMPT = """你是一个问答理解助手。你的任务是：
1. 解析用户输入，提取核心意图
2. 识别问题类型（fact_query|operation|analysis|clarification_needed）
3. 提取关键实体和参数
4. 如果模糊，识别需要澄清的点

输出JSON格式：
{
    "intent": "意图描述",
    "query_type": "类型",
    "entities": [],
    "parameters": {},
    "clarification": "澄清点",
    "processed_query": "处理后的问题"
}"""

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        query = str(input_data)
        context["original_query"] = query
        
        messages = self._build_messages(query, context)
        
        try:
            llm_res = await self._call_llm(messages, stream_callback, temperature=0.1)
            return self._process_llm_response(llm_res, context)
        except Exception as e:
            return AgentResult(AgentStatus.FAILURE, None, f"ParseFailed: {str(e)}")

    def _build_messages(self, query: str, context: Dict[str, Any]) -> List[Dict]:
        content = f"用户问题：{query}"
        if "session_files" in context:
            content += self._format_session_files(context["session_files"])
            
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

    def _format_session_files(self, files: List[Dict]) -> str:
        lines = ["\n\n用户提供的文件上下文："]
        for f in files:
            lines.append(f"文件: {f['name']}\n内容:\n{f['content']}\n")
        return "\n".join(lines)

    def _process_llm_response(self, llm_res: Dict[str, str], context: Dict[str, Any]) -> AgentResult:
        parsed = self._parse_json_response(llm_res["response"])
        thought = llm_res["thought"]
        
        if parsed.get("query_type") == "clarification_needed":
            return self._create_clarification_result(parsed, thought)
            
        context["parsed_query"] = parsed
        return AgentResult(
            AgentStatus.SUCCESS, 
            parsed, 
            f"Parsed: {parsed.get('intent')}", 
            next_agent="planning",
            thought_process=thought
        )

    def _create_clarification_result(self, parsed: Dict, thought: str) -> AgentResult:
        return AgentResult(
            status=AgentStatus.NEEDS_HUMAN,
            data={"question": parsed.get("clarification"), "original": parsed.get("processed_query")},
            message="ClarificationRequired",
            next_agent="qa",
            thought_process=thought
        )
