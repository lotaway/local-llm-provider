#!/usr/bin/env python3
"""
诊断 RAG 检索问题的脚本
检查：
1. 向量存储中是否有 ChatGPT 文档
2. BM25 索引是否包含关键词
3. 检索结果是什么
"""

import os
os.environ["LOG_LEVEL"] = "DEBUG"

from model_providers import LocalLLModel
from rag import LocalRAG

print("=" * 80)
print("RAG 检索诊断工具")
print("=" * 80)

# 初始化
llm = LocalLLModel()
rag = LocalRAG(llm, data_path="./docs", use_hybrid_search=True, use_reranking=True)

print("\n1️⃣ 初始化 RAG 链...")
rag.init_rag_chain()

print(f"\n2️⃣ 检查文档数量:")
print(f"   - all_documents: {len(rag.all_documents)} 个文档片段")

if rag.all_documents:
    print(f"\n3️⃣ 检查是否包含 'torch' 关键词:")
    torch_docs = [doc for doc in rag.all_documents if 'torch' in doc.page_content.lower()]
    print(f"   - 包含 'torch' 的文档: {len(torch_docs)} 个")
    
    if torch_docs:
        print(f"\n   📄 示例文档 (前3个):")
        for i, doc in enumerate(torch_docs[:3]):
            print(f"\n   文档 {i+1}:")
            print(f"   来源: {doc.metadata.get('source', 'unknown')}")
            print(f"   标题: {doc.metadata.get('title', 'N/A')}")
            print(f"   长度: {len(doc.page_content)} 字符")
            preview = doc.page_content[:200].replace('\n', ' ')
            print(f"   预览: {preview}...")

print(f"\n4️⃣ 测试检索 'torch':")
if rag.rag_chain:
    try:
        # 手动执行检索步骤 - 直接调用 retrieval_runnable
        # 从 init_rag_chain 中我们知道 chain 的第一步是 {"context": retrieval_runnable, "question": lambda x: x}
        # 我们需要直接测试检索器
        from rag import LocalRAG
        
        # 重新获取 vectorstore 和构建 retriever
        vectorstore = rag.get_or_create_vectorstore()
        from retrievers import HybridRetriever
        from typing import cast
        from langchain_milvus import Milvus
        
        vectorstore = cast(Milvus, vectorstore)
        if rag.use_hybrid_search and rag.all_documents:
            retriever = HybridRetriever(
                vectorstore=vectorstore,
                documents=rag.all_documents,
                vector_weight=0.7,
                bm25_weight=0.3,
                k=10
            )
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        
        docs = retriever.invoke("torch")
        
        print(f"   - 检索到 {len(docs)} 个文档")
        
        if docs:
            print(f"\n   📄 检索结果:")
            for i, doc in enumerate(docs):
                print(f"\n   结果 {i+1}:")
                print(f"   来源: {doc.metadata.get('source', 'unknown')}")
                print(f"   标题: {doc.metadata.get('title', 'N/A')}")
                print(f"   长度: {len(doc.page_content)} 字符")
                preview = doc.page_content[:300].replace('\n', ' ')
                print(f"   内容: {preview}...")
        else:
            print("   ❌ 没有检索到任何文档！")
    except Exception as e:
        print(f"   ❌ 检索测试失败: {e}")
        import traceback
        traceback.print_exc()

print(f"\n5️⃣ 测试完整 RAG 查询:")
answer = rag.generate_answer("torch")
print(f"   答案: {answer}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
