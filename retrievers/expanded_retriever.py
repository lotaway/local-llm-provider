class ExpandedRetriever:
    def __init__(self, vectorstore, search_kwargs={}):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs
    
    def __call__(self, inputs):
        question = inputs["question"]
        outline_context = inputs.get("outline_context", "")
        categories = inputs.get("categories", [])
        
        results = self.vectorstore.similarity_search(question, **self.search_kwargs)
        
        if outline_context:
            results.extend(self.vectorstore.similarity_search(f"Outline:{outline_context} {question}", **self.search_kwargs))
            
        for category in categories:
            results.extend(self.vectorstore.similarity_search(f"Category:{category}", k=1))
        
        seen, unique = set(), []
        for doc in results:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                unique.append(doc)
                
        return unique[:5]