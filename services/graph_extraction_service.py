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
- **丰富属性**：在 `properties` 中记录描述。
- **输出格式**：仅输出 JSON。

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
                max_new_tokens=1536
            )
            clean_response = self.llm.extract_after_think(response)
            
            logger.info(f"Extraction complete in {time.time() - start_time:.2f}s")
            
            json_text = self._clean_json_response(clean_response)
            json_text = self._clean_json_response(clean_response)
            try:
                data = json.loads(json_text)
            except json.JSONDecodeError:
                try:
                    fixed_json = re.sub(r',\s*([\]}])', r'\1', json_text)
                    
                    try:
                        data = json.loads(fixed_json)
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
                    entity = Entity(**e_data)
                    extracted_entities.append(entity)
                    logger.debug(f"Extracted entity: {entity.canonical_name} ({entity.type})")
                except Exception as ve:
                    logger.warning(f"Skipping invalid entity: {ve}")

            extracted_relations = []
            for r_data in data.get("relations", []):
                try:
                    relation = Relation(**r_data)
                    extracted_relations.append(relation)
                    logger.debug(f"Extracted relation: {relation.from_entity_id} -> {relation.relation_type} -> {relation.to_entity_id}")
                except Exception as ve:
                    logger.warning(f"Skipping invalid relation: {ve}")
                    
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
