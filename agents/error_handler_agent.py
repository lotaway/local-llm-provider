"""Error Handler Agent - Automated anomaly attribution and resolution"""

import json
import logging
from typing import Any, Dict, List, Optional
from .agent_base import BaseAgent, AgentResult, AgentStatus
from .error_utils import EvidenceExtractor, SystemSnapshot

logger = logging.getLogger(__name__)

class ErrorHandlerAgent(BaseAgent):
    """
    Agent for automated anomaly attribution and resolution.
    Follows the 5-layer architecture:
    1. Evidence Extraction (via Utils)
    2. System Snapshot (via Utils)
    3. Root Cause Analysis (Matching against Taxonomy)
    4. Hypothesis-Verification
    5. Action Decision
    """

    ROOT_CAUSE_TAXONOMY = {
        "LLM_TIMEOUT": "LLM service is slow or unresponsive.",
        "LLM_PARSE_ERROR": "Failed to parse LLM structured response.",
        "TOOL_EXECUTION_ERROR": "An external tool (MCP/RAG) failed during execution.",
        "CONTEXT_OVERFLOW": "Context window exceeded maximum limit.",
        "PERMISSION_DENIED": "Operation rejected due to insufficient permissions.",
        "INVALID_PLAN": "The generated plan is logically impossible or cyclical.",
        "RESOURCE_EXHAUSTED": "Local resources (memory/disk) are insufficient.",
        "UNKNOWN_ERROR": "Uncategorized system error."
    }

    SYSTEM_PROMPT = """你是一个“自动化异常归因与处置”专家。你的任务是将系统抛出的原始错误转化为可落地的根因分析和处置方案。

你的工作原则：
1. 模型不负责删证据，只负责解释证据。
2. 自动化决策 = 归因置信度 × Action 风险。
3. 如果无法通过工具证伪，必须降级为人确认。

根因分类库 (Taxonomy):
{taxonomy}

输出格式 (JSON):
{{
    "root_cause_analysis": {{
        "primary_cause": "TAXONOMY_KEY",
        "confidence": 0.0-1.0,
        "explanation": "原因解释",
        "supporting_evidence": ["证据1", "证据2"],
        "counter_evidence": []
    }},
    "verification_step": {{
        "hypothesis": "待验证的假设",
        "tool_to_verify": "具体的工具或方法",
        "is_verified": true/false
    }},
    "proposed_action": {{
        "action": "具体的处置动作",
        "risk_score": 0.0-1.0,
        "blast_radius": "impact_range",
        "rollback_plan": "回滚方案",
        "confidence_threshold_met": true/false
    }},
    "decision": "AUTOMATIC | MANUAL"
}}

决策规则：
- confidence >= 0.95 && risk <= 0.2 -> AUTOMATIC
- confidence >= 0.9 && risk <= 0.1 -> AUTOMATIC
- 其他情况 -> MANUAL
"""

    def __init__(self, llm_model, name: str = None):
        super().__init__(llm_model, name)
        self.taxonomy = json.dumps(self.ROOT_CAUSE_TAXONOMY, indent=2, ensure_ascii=False)

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        """
        Execute error handling workflow.
        input_data should contain: {'exception': exception_object, 'agent_name': name}
        """
        exception = input_data.get("exception")
        original_agent = input_data.get("agent_name")
        
        # 1. & 2. Evidence Gathering (Non-LLM)
        evidence = EvidenceExtractor.extract_from_exception(exception, original_agent, context)
        snapshot = SystemSnapshot.capture(context)
        
        # 3. & 4. Root Cause Matching & Hypothesis (LLM)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(taxonomy=self.taxonomy)},
            {
                "role": "user",
                "content": f"证据包: {json.dumps(evidence, ensure_ascii=False)}\n系统快照: {json.dumps(snapshot, ensure_ascii=False)}"
            }
        ]
        
        try:
            response = await self._call_llm(
                messages,
                stream_callback=stream_callback,
                temperature=0.0
            )
            analysis = self._parse_json_response(response)
            
            # 5. Decision Layer
            decision = analysis.get("decision", "MANUAL")
            confidence = analysis.get("root_cause_analysis", {}).get("confidence", 0.0)
            risk = analysis.get("proposed_action", {}).get("risk_score", 1.0)
            
            # Validate automatic execution threshold
            is_auto = False
            if confidence >= 0.95 and risk <= 0.2:
                is_auto = True
            elif confidence >= 0.9 and risk <= 0.1:
                is_auto = True
                
            if is_auto and decision == "AUTOMATIC":
                # Apply fix to context if applicable, or pass as data
                context["last_error_fix"] = analysis.get("proposed_action")
                return AgentResult(
                    status=AgentStatus.CONTINUE,
                    data=analysis.get("proposed_action"),
                    message=f"已自动处理异常并恢复: {analysis.get('root_cause_analysis', {}).get('primary_cause')}",
                    next_agent="planning", # Resume from planning with fix in context
                    metadata=analysis
                )
            else:
                return AgentResult(
                    status=AgentStatus.NEEDS_HUMAN,
                    data=analysis,
                    message=f"无法自动处理该异常，已转由人工决策。置信度: {confidence}, 风险: {risk}",
                    next_agent="planning",
                    metadata=analysis
                )
                
        except Exception as e:
            self.logger.error(f"ErrorHandlerAgent failed: {e}")
            return AgentResult(
                status=AgentStatus.NEEDS_HUMAN,
                data={"original_error": str(exception), "handler_error": str(e)},
                message="异常处理代理执行失败，请人工介入",
                next_agent="planning"
            )
