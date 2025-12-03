from typing import List, Dict, Any
from langchain_core.documents import Document
from .base_loader import BaseChatLoader


class DeepSeekLoader(BaseChatLoader):
    def check(self, data: Any):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "mapping" in data[0] and "inserted_at" in data[0]:
            return True
        return False
    
    
    def load(self, data: list, source_file: str) -> List[Document]:
        """解析 DeepSeek 导出的 JSON 数据"""
        docs = []
        for conversation in data:
            title = conversation.get("title", "Untitled Conversation")
            mapping = conversation.get("mapping", {})
            
            # 寻找根节点
            root_id = None
            for node_id, node in mapping.items():
                if node.get("parent") is None:
                    root_id = node_id
                    break
            
            if not root_id:
                continue
                
            # 遍历对话树
            current_id = root_id
            messages = []
            
            while current_id:
                node = mapping.get(current_id)
                if not node:
                    break
                
                message = node.get("message")
                if message:
                    fragments = message.get("fragments", [])
                    for fragment in fragments:
                        frag_type = fragment.get("type")
                        content = fragment.get("content", "")
                        
                        role = None
                        if frag_type == "REQUEST":
                            role = "user"
                        elif frag_type == "RESPONSE":
                            role = "assistant"
                        
                        if role and content:
                            messages.append({"role": role, "content": content})

                # 移动到下一个节点
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
                
                # 如果是 User 消息，且当前 buffer 中已有 Assistant 消息，说明上一轮对话结束，先保存
                if role == "user":
                    if current_doc_messages and current_doc_messages[-1]["role"] == "assistant":
                        docs.append(self._create_chat_doc(title, current_doc_messages, source_file))
                        current_doc_messages = []
                
                current_doc_messages.append(msg)
            
            # 处理剩余的消息
            if current_doc_messages:
                docs.append(self._create_chat_doc(title, current_doc_messages, source_file))
                current_doc_messages = [] 
        return docs

    def _create_chat_doc(self, title, messages, source):
        """构建 Document 对象"""
        content_parts = [f"Title: {title}"]
        for msg in messages:
            role_prefix = "Question" if msg["role"] == "user" else "Answer"
            content_parts.append(f"{role_prefix}: {msg['content']}")
            
        full_content = "\n\n".join(content_parts)
        return Document(page_content=full_content, metadata={"source": source, "title": title})