import logging
import asyncio
from typing import Dict, Any, List, Optional
from agents.agent_base import BaseAgent, AgentResult, AgentStatus

logger = logging.getLogger("service.skill_researcher")

class SkillResearcher:
    """Service to research and plan new skills based on feedback or gaps."""
    
    def __init__(self, llm_model):
        self.llm = llm_model

    async def research_missing_skill(self, task_description: str, error_context: str) -> Dict[str, Any]:
        """Research a solution for a missing skill or tool."""
        logger.info(f"Researching missing skill for: {task_description}")
        
        prompt = f"""你是一个高级AI技能研究员。系统在执行以下任务时由于缺少相应工具而失败：
任务：{task_description}
错误：{error_context}

请研究并提供一个新技能的设计方案（Skill IR）。
输出JSON格式：
{{
    "skill_name": "新技能名称",
    "purpose": "技能目标",
    "tools": [
        {{
            "name": "工具名称",
            "description": "工具描述",
            "parameters": {{ "param1": "type", ... }},
            "logic_pseudo_code": "核心逻辑伪代码"
        }}
    ],
    "research_findings": "研究结论与参考资料"
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        try:
            # Use chat_at_once or similar
            raw_res = await self.llm.chat_at_once(messages)
            # Need to parse JSON. I'll use a simple parser here.
            import json
            import re
            json_match = re.search(r"\{.*\}", raw_res, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"error": "Failed to parse researcher response"}
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {"error": str(e)}

class SkillCreatorBridge:
    """Bridge to execute the skill-creator and deploy the skill."""
    
    def __init__(self, registry):
        self.registry = registry

    async def create_and_deploy(self, skill_ir: Dict[str, Any]) -> bool:
        """Call the skill-creator (agent or script) to generate code and deploy."""
        logger.info(f"Deploying new skill: {skill_ir.get('skill_name')}")
        
        # In a real implementation, this would call an Agent that writes Python code
        # to the skills/generated/ directory and then calls self.registry.refresh_generated_skills()
        
        # For now, we simulate the deployment
        try:
            # 1. Generate code (MOCK)
            # 2. Save to skills/generated/ (MOCK)
            # 3. Reload registry
            # self.registry.refresh_generated_skills()
            return True
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
