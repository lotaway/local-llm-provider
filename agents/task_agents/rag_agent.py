"""RAG Task Agent - RAG-based retrieval and generation"""

from typing import Any, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.agent_base import BaseAgent, AgentResult, AgentStatus


class RAGTaskAgent(BaseAgent):
    """Agent for RAG-based retrieval and generation"""
    
    def __init__(self, llm_model, rag_instance=None, name: str = None):
        """
        Initialize RAG task agent
        
        Args:
            llm_model: LocalLLModel instance
            rag_instance: LocalRAG instance (optional, will be created if needed)
            name: Agent name
        """
        super().__init__(llm_model, name)
        self.rag = rag_instance
    
    def execute(self, input_data: Any, context: Dict[str, Any], stream_callback=None) -> AgentResult:
        """
        Execute RAG task
        
        Args:
            input_data: Task definition
            context: Runtime context
            stream_callback: Optional callback for streaming LLM outputs
            
        Returns:
            AgentResult with RAG response
        """
        task = input_data if isinstance(input_data, dict) else {}
        task_description = task.get("description", "")
        original_query = context.get("original_query", "")
        
        # Construct query for RAG
        query = f"{original_query}\n任务：{task_description}" if task_description else original_query
        
        try:
            # Initialize RAG if not already done
            if self.rag is None:
                from rag import LocalRAG
                data_path = os.getenv("DATA_PATH", "./docs")
                self.rag = LocalRAG(self.llm, data_path=data_path)
            
            # Execute RAG query
            answer = self.rag.generate_answer(query)
            
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=answer,
                message="RAG任务完成",
                next_agent="verification"
            )
            
        except Exception as e:
            self.logger.error(f"RAG task failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message=f"RAG任务失败: {str(e)}"
            )
