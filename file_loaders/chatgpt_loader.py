from typing import List, Dict, Any
from langchain_core.documents import Document
from .base_loader import BaseChatLoader

class ChatGPTLoader(BaseChatLoader):
    def check(self, data: Any):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "mapping" in data[0] and "create_time" in data[0]:
            return True
        return False
    
    def load(self, data: list, source_file: str) -> List[Document]:
        docs = []
        for conversation in data:
            title = conversation.get("title", "Untitled Conversation")
            mapping = conversation.get("mapping", {})
            
            root_id = None
            for node_id, node in mapping.items():
                if node.get("parent") is None:
                    root_id = node_id
                    break
            
            if not root_id:
                continue
                
            current_id = root_id
            messages = []
            
            while current_id:
                node = mapping.get(current_id)
                if not node:
                    break
                
                message = node.get("message")
                if message:
                    author = message.get("author", {})
                    role = author.get("role")
                    content_dict = message.get("content", {})
                    parts = content_dict.get("parts", [])
                    
                    text_content = ""
                    if parts:
                        text_content = "".join([str(p) for p in parts if p is not None])
                    
                    if text_content and role in ["user", "assistant"]:
                        messages.append({"role": role, "content": text_content})
                
                children = node.get("children", [])
                if children:
                    current_id = children[0]
                else:
                    current_id = None
            
            if not messages:
                continue

            current_doc_messages = []
            
            for i, msg in enumerate(messages):
                role = msg["role"]
                
                if role == "user":
                    if current_doc_messages and current_doc_messages[-1]["role"] == "assistant":
                        docs.append(self._create_chat_doc(title, current_doc_messages, source_file))
                        current_doc_messages = []
                
                current_doc_messages.append(msg)
            
            if current_doc_messages:
                docs.append(self._create_chat_doc(title, current_doc_messages, source_file))
                current_doc_messages = [] 
        return docs

    def _create_chat_doc(self, title, messages, source):
        content_parts = [f"Title: {title}"]
        for msg in messages:
            role_prefix = "Question" if msg["role"] == "user" else "Answer"
            content_parts.append(f"{role_prefix}: {msg['content']}")
            
        full_content = "\n\n".join(content_parts)
        return Document(page_content=full_content, metadata={"source": source, "title": title})

