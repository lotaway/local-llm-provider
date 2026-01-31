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
    EXTRACTION_PROMPT = """你是一个专业的语义知识图谱专家。
你的任务是：深入理解提供的文本，并将其中的关键实体和核心关系提取出来。

### 1. 实体类型 (Entities)：
请根据语义将实体归入以下最合适的类别：
- **Document**: 文档、文件、协议说明
- **Concept**: 核心概念、抽象理论、产品特性（如：“PBT材质”、“热升华工艺”）
- **Code/API**: 函数、变量、程序接口、技术协议
- **Organization**: 公司、团队、品牌、机构
- **Person**: 具体的人物
- **Model/Product**: 具体型号、硬件产品（如：“红武士键帽”、“Q2键盘”）

### 2. 关系类型 (Relations)：
提取能反映逻辑联系的关系：
- **uses**: 使用/应用
- **depends_on**: 依赖/前提
- **implements**: 实现/体现了
- **describes**: 描述/定义了
- **part_of / contains**: 包含/组成关系

### 3. 操作要求：
- **稳定 ID**：`stable_id` 请使用 `类型_名称小写化`。
- **精简属性**：在 `properties` 中**仅记录最关键的 1-2 个属性**（如：材质、型号），避免冗长描述。
- **输出格式**：仅输出 JSON，不要输出任何解释说明。

### 待处理文本：
{text}
"""

    def __init__(self, llm: LocalLLModel):
        self.llm = llm

    async def extract_graph(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        prompt = self.EXTRACTION_PROMPT.replace("{text}", text)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            logger.info(f"Extracting semantic graph (text len: {len(text)})...")
            start_time = time.time()
            response = await self.llm.chat_at_once(
                messages, 
                temperature=0.2,
                top_p=0.95,
                max_new_tokens=3072
            )
            clean_response = self.llm.extract_after_think(response)
            
            logger.info(f"Extraction complete in {time.time() - start_time:.2f}s")
            
            json_text = self._clean_json_response(clean_response)
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError as parse_error:
                try:
                    fixed_json = re.sub(r',\s*([\]}])', r'\1', json_text)
                    if not fixed_json.endswith("}") and not fixed_json.endswith("]"):
                        last_complete_obj = fixed_json.rfind('},')
                        if last_complete_obj > 0:
                            fixed_json = fixed_json[:last_complete_obj + 1]
                        
                        open_braces = fixed_json.count("{") - fixed_json.count("}")
                        open_brackets = fixed_json.count("[") - fixed_json.count("]")
                        
                        fixed_json += ']' * open_brackets
                        fixed_json += '}' * open_braces

                    try:
                        data = json.loads(fixed_json)
                        logger.warning(f"JSON was truncated but successfully repaired (original error: {parse_error})")
                    except json.JSONDecodeError:
                        if "'" in fixed_json and '"' not in fixed_json:
                            fixed_json = fixed_json.replace("'", '"')
                        data = json.loads(fixed_json)
                except Exception as e:
                    logger.error(f"JSON Parse Error after repair attempt: {e}")
                    logger.error(f"Cleaned text snippet: {json_text[:200]}...")
                    raise

            extracted_entities = []
            for e_data in data.get("entities", []):
                try:
                    if "canonical_name" not in e_data:
                        if "name" in e_data:
                            e_data["canonical_name"] = e_data["name"]
                        elif "stable_id" in e_data and "_" in e_data["stable_id"]:
                            parts = e_data["stable_id"].split("_", 1)
                            if len(parts) > 1:
                                e_data["canonical_name"] = parts[1]
                    
                    if "type" not in e_data and "stable_id" in e_data:
                         parts = e_data["stable_id"].split("_", 1)
                         if len(parts) > 0:
                             e_data["type"] = parts[0].capitalize()

                    entity = Entity(**e_data)
                    extracted_entities.append(entity)
                    logger.debug(f"Extracted entity: {entity.canonical_name} ({entity.type})")
                except Exception as ve:
                    logger.warning(f"Skipping invalid entity: {ve} Data: {e_data}")

            extracted_relations = []
            for r_data in data.get("relations", []):
                try:
                    if "source" in r_data and "from_entity_id" not in r_data:
                        r_data["from_entity_id"] = r_data["source"]
                    elif "from" in r_data and "from_entity_id" not in r_data:
                        r_data["from_entity_id"] = r_data["from"]
                    elif "head" in r_data and "from_entity_id" not in r_data:
                        r_data["from_entity_id"] = r_data["head"]
                    
                    if "target" in r_data and "to_entity_id" not in r_data:
                        r_data["to_entity_id"] = r_data["target"]
                    elif "to" in r_data and "to_entity_id" not in r_data:
                        r_data["to_entity_id"] = r_data["to"]
                    elif "tail" in r_data and "to_entity_id" not in r_data:
                        r_data["to_entity_id"] = r_data["tail"]

                    if "type" in r_data and "relation_type" not in r_data:
                        r_data["relation_type"] = r_data["type"]
                    elif "relation" in r_data and "relation_type" not in r_data:
                        r_data["relation_type"] = r_data["relation"]

                    relation = Relation(**r_data)
                    extracted_relations.append(relation)
                    logger.debug(f"Extracted relation: {relation.from_entity_id} -> {relation.relation_type} -> {relation.to_entity_id}")
                except Exception as ve:
                    logger.warning(f"Skipping invalid relation: {ve} Data: {r_data}")
                    
            return extracted_entities, extracted_relations

        except Exception as e:
            logger.error(f"Failed to extract graph using main LLM: {e}")
            return [], []

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
        if end > start:
            return text[start:end]
        return text.strip()
