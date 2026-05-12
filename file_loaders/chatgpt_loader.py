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
            title = conversation.get("title", "Untitled")
            messages = self._extract_messages(conversation.get("mapping", {}))
            if messages:
                docs.extend(self._segment_messages(title, messages, source_file))
        return docs

    def _extract_messages(self, mapping: dict) -> list:
        root_id = next((nid for nid, n in mapping.items() if not n.get("parent")), None)
        if not root_id: return []
        
        msgs, curr = [], root_id
        while curr:
            node = mapping.get(curr)
            if not node: break
            msg = node.get("message")
            if msg and msg.get("author", {}).get("role") in ["user", "assistant"]:
                text = "".join(str(p) for p in msg.get("content", {}).get("parts", []) if p)
                if text: msgs.append({"role": msg["author"]["role"], "content": text})
            curr = node.get("children", [None])[0]
        return msgs

    def _segment_messages(self, title: str, messages: list, source: str) -> List[Document]:
        docs, current = [], []
        for msg in messages:
            if msg["role"] == "user" and current and current[-1]["role"] == "assistant":
                docs.append(self._create_chat_doc(title, current, source))
                current = []
            current.append(msg)
        if current: docs.append(self._create_chat_doc(title, current, source))
        return docs

    def _create_chat_doc(self, title, messages, source):
        content_parts = [f"Title: {title}"]
        for msg in messages:
            role_prefix = "Question" if msg["role"] == "user" else "Answer"
            content_parts.append(f"{role_prefix}: {msg['content']}")
            
        full_content = "\n\n".join(content_parts)
        return Document(page_content=full_content, metadata={"source": source, "title": title})

