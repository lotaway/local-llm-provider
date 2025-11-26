"""Q&A Agent - Handles user interaction and query understanding"""

from typing import Any, Dict
from .agent_base import BaseAgent, AgentResult, AgentStatus


class QAAgent(BaseAgent):
    """Agent for parsing and understanding user queries"""
    
    SYSTEM_PROMPT = """你是一个问答理解助手。你的任务是：
1. 解析用户输入，提取核心意图
2. 识别问题类型（事实查询、操作请求、分析任务等）
3. 提取关键实体和参数
4. 如果问题模糊，识别需要澄清的点

输出JSON格式：
{
    "intent": "用户意图描述",
    "query_type": "fact_query|operation|analysis|clarification_needed",
    "entities": ["实体1", "实体2"],
    "parameters": {"参数名": "参数值"},
    "clarification": "如果需要澄清，说明需要澄清什么",
    "processed_query": "处理后的清晰问题"
}"""
    
    def execute(self, input_data: Any, context: Dict[str, Any], stream_callback=None) -> AgentResult:
        """
        Parse and understand user query
        
        Args:
            input_data: User query string
            context: Runtime context
            stream_callback: Optional callback for streaming LLM outputs
            
        Returns:
            AgentResult with parsed query information
        """
        query = str(input_data)
        
        # Store original query in context
        context["original_query"] = query
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"用户问题：{query}"}
        ]
        
        try:
            response = self._call_llm(messages, stream_callback=stream_callback, temperature=0.1, max_new_tokens=1000)
            parsed = self._parse_json_response(response)
            
            # Check if clarification needed
            if parsed.get("query_type") == "clarification_needed":
                return AgentResult(
                    status=AgentStatus.NEEDS_HUMAN,
                    data={
                        "question": parsed.get("clarification", "请提供更多信息"),
                        "original_query": query
                    },
                    message="需要用户澄清问题",
                    next_agent="qa"
                )
            
            # Store parsed query in context
            context["parsed_query"] = parsed
            
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=parsed,
                message=f"成功解析查询: {parsed.get('intent', 'unknown')}",
                next_agent="planning"
            )
            
        except Exception as e:
            self.logger.error(f"Query parsing failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message=f"查询解析失败: {str(e)}"
            )
