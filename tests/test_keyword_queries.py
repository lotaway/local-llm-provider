#!/usr/bin/env python3
"""测试关键词查询的改进效果"""

import os
os.environ["LOG_LEVEL"] = "WARNING"  # 减少日志输出

from model_providers import LocalLLModel
from rag import LocalRAG

print("=" * 80)
print("测试关键词查询改进")
print("=" * 80)

llm = LocalLLModel()
rag = LocalRAG(llm, data_path="./docs", use_hybrid_search=True, use_reranking=True)

print("\n初始化 RAG 系统...")
rag.init_rag_chain()
print(f"✅ 已加载 {len(rag.all_documents)} 个文档片段")

# 测试不同类型的查询
test_cases = [
    {
        "type": "关键词查询",
        "queries": [
            "torch",
            "torch-directml",
            "Python 3.11"
        ]
    },
    {
        "type": "完整问题",
        "queries": [
            "torch-directml 需要什么 Python 版本？",
            "如何安装 torch-directml？"
        ]
    }
]

for test_case in test_cases:
    print(f"\n{'='*80}")
    print(f"📋 测试类型: {test_case['type']}")
    print('='*80)
    
    for query in test_case['queries']:
        print(f"\n🔍 查询: {query}")
        print("-" * 80)
        try:
            answer = rag.generate_answer(query)
            print(f"💬 答案:\n{answer}")
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        print()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
