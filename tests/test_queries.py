#!/usr/bin/env python3
"""测试不同查询方式的效果"""

import os
os.environ["LOG_LEVEL"] = "INFO"

from model_providers import LocalLLModel
from rag import LocalRAG

print("=" * 80)
print("测试不同查询方式")
print("=" * 80)

llm = LocalLLModel()
rag = LocalRAG(llm, use_hybrid_search=True, use_reranking=True)
rag.init_rag_chain()

# 测试不同的查询
queries = [
    "torch",
    "torch-directml",
    "torch-directml 需要什么 Python 版本",
    "torch-directml 2.3.0 的要求",
]

for query in queries:
    print(f"\n{'='*80}")
    print(f"查询: {query}")
    print('='*80)
    try:
        answer = rag.generate_answer(query)
        print(f"答案: {answer}")
    except Exception as e:
        print(f"错误: {e}")
