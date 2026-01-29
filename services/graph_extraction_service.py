import json
import logging
import re
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_providers import LocalLLModel
from schemas.graph import Entity, Relation
import os

logger = logging.getLogger(__name__)


class GraphExtractionService:
    EXTRACTION_PROMPT = """你是一个知识图谱结构抽取器。
你的职责是：仅从文本中抽取“明确出现的实体和关系”，不得进行推理或补全。

### 实体类型（封闭集合）：
只能从以下类型中选择：
Document, Concept, Code, API, Event, Model, Person, Organization

如果无法准确分类，使用 Concept，不得创建新类型。

---

### 关系类型（封闭集合）：
uses, depends_on, implements, describes, references, member_of, part_of, contains, contradicts

规则：
- 只有当文本中明确表达该关系时才可抽取
- 如果只是“提到 / 说明 / 举例”，使用 references 或 describes
- 不得基于常识或经验推断关系

---

### 禁止行为（非常重要）：
- 不得引入文本中未明确出现的实体
- 不得推断隐含关系
- 不得合并不同名称的实体
- 不得修改或解释 Schema 含义

---

### 实体 ID 规则：
- stable_id = Type_规范化名称
- 规范化规则：
  - 全小写
  - 去除空格与特殊字符
- 同一文本内相同名称必须生成相同 ID
- 不得基于上下文语义生成 ID

---

### 关系置信度规则：
- 明确陈述的事实关系：0.8–1.0
- 间接描述但无歧义：0.6–0.8
- 仅上下文提及：≤ 0.5

---

### 输出要求：
- 仅返回 JSON
- 所有关系必须引用已定义的实体 stable_id
- 不输出解释性文本

### 输出格式：
{
    "entities": [
        {"stable_id": "Doc_README_md", "type": "Document", "canonical_name": "README.md", "properties": {"version": "1.0"}}
    ],
    "relations": [
        {"from_entity_id": "Doc_README_md", "to_entity_id": "Concept_GraphRAG", "relation_type": "describes", "confidence": 0.9, "properties": {}}
    ]
}

### 待处理文本：
{text}
"""

    def __init__(self, llm: LocalLLModel):
        self.llm = llm
        self.small_model = None
        self.tokenizer = None

    def _ensure_model_loaded(self):
        if self.small_model is not None:
            return

        model_name = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        logger.info(f"Loading graph extraction model {model_name} on {device}...")

        transformers_kwargs = {}
        if device == "cuda":
            transformers_kwargs["device_map"] = "auto"
            transformers_kwargs["torch_dtype"] = torch.float16
        else:
            transformers_kwargs["device_map"] = "cpu"
            transformers_kwargs["torch_dtype"] = torch.float32

        transformers_kwargs["trust_remote_code"] = True
        transformers_kwargs["cache_dir"] = os.getenv("CACHE_PATH", "./cache")

        self.small_model = AutoModelForCausalLM.from_pretrained(
            model_name, **transformers_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=os.getenv("CACHE_PATH", "./cache")
        )
        self.small_model.eval()
        logger.info("Graph extraction model loaded.")

    async def extract_graph(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        self._ensure_model_loaded()
        prompt = self.EXTRACTION_PROMPT.replace("{text}", text)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.small_model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    top_k=40,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            json_text = self._clean_json_response(response)

            data = json.loads(json_text)

            extracted_entities = []
            for e_data in data.get("entities", []):
                try:
                    extracted_entities.append(Entity(**e_data))
                except Exception as ve:
                    logger.warning(f"Skipping invalid entity: {ve}")

            extracted_relations = []
            for r_data in data.get("relations", []):
                try:
                    extracted_relations.append(Relation(**r_data))
                except Exception as ve:
                    logger.warning(f"Skipping invalid relation: {ve}")
            return extracted_entities, extracted_relations
        except Exception as e:
            logger.error(f"Failed to extract graph from text: {e}")
            return [], []

    def _clean_json_response(self, text: str) -> str:
        if "```json" in text:
            match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            return text[start:end]

        return text.strip()
