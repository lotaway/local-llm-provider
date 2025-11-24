class ExpandedRetriever:
    def __init__(self, vectorstore, search_kwargs = {}):
        self.vectorstore = vectorstore
    
    def __call__(self, inputs):
        question = inputs["question"]
        outline_context = inputs.get("outline_context", "")
        categories = inputs.get("categories", [])
        all_results = []
        direct_results = self.vectorstore.similarity_search(question, **self.search_kwargs)
        all_results.extend(direct_results)
        if outline_context:
            outline_query = f"目录:{outline_context} {question}"
            outline_results = self.vectorstore.similarity_search(outline_query, **self.search_kwargs)
            all_results.extend(outline_results)
        for category in categories:
            category_results = self.vectorstore.similarity_search(
                f"分类:{category}", k=1
            )
            all_results.extend(category_results)
        
        # 去重并返回
        seen_ids = set()
        unique_results = []
        for doc in all_results:
            doc_id = doc.page_content[:100]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)
        
        return unique_results[:5]


# def test():
#     vectorstore = Milvus
#     expanded_retriever = ExpandedRetriever(vectorstore)
#     rag_chain = (
#         {"context": expanded_retriever, 
#          "question": lambda x: x["question"]}
#         | prompt_str
#         | format_messages_runnable
#         | chat_runnable
#         | StrOutputParser()
#     )
#     result = rag_chain.invoke({
#         "question": "API认证", 
#         "outline_context": "用户管理/权限系统",
#         "categories": ["安全", "用户认证", "权限管理"]
#     })
#     print(result)