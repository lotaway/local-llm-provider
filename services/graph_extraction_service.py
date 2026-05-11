import json
import logging
import re
import os
import asyncio
import time
from typing import List, Dict, Any, Tuple
from model_providers import LocalLLModel
from schemas.graph import Entity, Relation

logger = logging.getLogger(__name__)


class GraphExtractionService:
    EXTRACTION_PROMPT = """You are a professional semantic knowledge graph expert.
Your task is to understand the provided text and extract key entities and core relations.

### 1. Entities:
Categorize entities into the most appropriate category:
- **Document**: Files, protocols, documentation.
- **Concept**: Core ideas, theories, product features.
- **Code/API**: Functions, variables, interfaces, technical protocols.
- **Organization**: Companies, teams, brands, agencies.
- **Person**: Specific individuals.
- **Model/Product**: Specific models, hardware products.

### 2. Relations:
Extract logical connections:
- **uses**: Application or usage.
- **depends_on**: Dependency or prerequisite.
- **implements**: Implementation of a concept.
- **describes**: Description or definition.
- **part_of / contains**: Composition or containment.

### 3. Requirements:
- **Stable ID**: Use `type_name_lowercase` for `stable_id`.
- **Properties**: Record only 1-2 critical properties.
- **Format**: Output JSON only. No explanations.

### Text:
{text}
"""

    def __init__(self, llm: LocalLLModel):
        self.llm = llm

    async def extract_graph(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        try:
            start_time = time.time()
            prompt = self.EXTRACTION_PROMPT.format(text=text)
            response = await self.llm.chat_at_once(
                [{"role": "user", "content": prompt}],
                temperature=0.2, top_p=0.95, max_new_tokens=3072
            )
            clean_res = self.llm.extract_after_think(response)
            logger.info(f"Extraction complete in {time.time() - start_time:.2f}s")
            return self._parse_json_to_graph(clean_res)
        except Exception as e:
            logger.error(f"Graph extraction failed: {e}")
            return [], []

    def _parse_json_to_graph(self, response: str) -> Tuple[List[Entity], List[Relation]]:
        json_text = self._clean_json_response(response)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = self._attempt_json_repair(json_text)
        
        entities = self._map_entities(data.get("entities", []))
        relations = self._map_relations(data.get("relations", []))
        return entities, relations

    def _attempt_json_repair(self, json_text: str) -> Dict:
        fixed = re.sub(r',\s*([\]}])', r'\1', json_text)
        if not (fixed.endswith("}") or fixed.endswith("]")):
            fixed = self._close_open_structures(fixed)
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            if "'" in fixed and '"' not in fixed:
                fixed = fixed.replace("'", '"')
            return json.loads(fixed)

    def _close_open_structures(self, text: str) -> str:
        last_obj = text.rfind('},')
        if last_obj > 0:
            text = text[:last_obj + 1]
        
        text += ']' * (text.count("[") - text.count("]"))
        text += '}' * (text.count("{") - text.count("}"))
        return text

    def _map_entities(self, entity_data: List[Dict]) -> List[Entity]:
        entities = []
        for data in entity_data:
            self._fill_entity_defaults(data)
            try:
                entities.append(Entity(**data))
            except Exception as e:
                logger.warning(f"Invalid entity data: {e}")
        return entities

    def _fill_entity_defaults(self, data: Dict):
        if "canonical_name" not in data:
            data["canonical_name"] = data.get("name") or data.get("stable_id", "").split("_", 1)[-1]
        if "type" not in data and "stable_id" in data:
            data["type"] = data["stable_id"].split("_", 1)[0].capitalize()

    def _map_relations(self, relation_data: List[Dict]) -> List[Relation]:
        relations = []
        for data in relation_data:
            self._fill_relation_fields(data)
            try:
                relations.append(Relation(**data))
            except Exception as e:
                logger.warning(f"Invalid relation data: {e}")
        return relations

    def _fill_relation_fields(self, data: Dict):
        if "from_entity_id" not in data:
            data["from_entity_id"] = data.get("source") or data.get("from") or data.get("head")
        if "to_entity_id" not in data:
            data["to_entity_id"] = data.get("target") or data.get("to") or data.get("tail")
        if "relation_type" not in data:
            data["relation_type"] = data.get("type") or data.get("relation")

    def _clean_json_response(self, text: str) -> str:
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        start = text.find("{")
        if start == -1:
            return text.strip()
            
        brace_count = 0
        in_string = False
        escape = False
        
        for i in range(start, len(text)):
            char = text[i]
            if char == '"' and not escape:
                in_string = not in_string
            elif char == '\\' and in_string:
                escape = not escape
                continue
            elif not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start : i + 1]
            escape = False
        
        end = text.rfind("}") + 1
        return text[start:end] if end > start else text.strip()
