import logging
import asyncio
from typing import Dict, Any, List, Optional
from agents.agent_base import BaseAgent, AgentResult, AgentStatus

logger = logging.getLogger("service.skill_researcher")

class SkillResearcher:
    def __init__(self, llm_model):
        self.llm = llm_model

    async def research_missing_skill(self, task_description: str, error_context: str) -> Dict[str, Any]:
        logger.info(f"Researching missing skill for: {task_description}")
        
        prompt = f"""You are a Senior AI Skill Researcher. The system failed to execute the following task due to missing tools:
Task: {task_description}
Error: {error_context}

Please research and provide a design for a new skill (Skill IR).
Output JSON format:
{{
    "skill_name": "New skill name",
    "purpose": "Goal of the skill",
    "tools": [
        {{
            "name": "Tool name",
            "description": "Tool description",
            "parameters": {{ "param1": "type", ... }},
            "logic_pseudo_code": "Core logic pseudo-code"
        }}
    ],
    "research_findings": "Research conclusions and references"
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        try:
            raw_res = await self.llm.chat_at_once(messages)
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
    def __init__(self, registry):
        self.registry = registry

    async def create_and_deploy(self, skill_ir: Dict[str, Any]) -> bool:
        logger.info(f"Deploying new skill: {skill_ir.get('skill_name')}")
        
        try:
            return True
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
