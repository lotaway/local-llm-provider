from typing import List, Dict, Any
from langchain_core.documents import Document
from .base_loader import BaseChatLoader

class ChatGPTLoader(BaseChatLoader):
    def load(self, data: list, source_file: str) -> List[Document]:
        """解析 ChatGPT 导出的 JSON 数据"""
        docs = []
        for conversation in data:
            title = conversation.get("title", "Untitled Conversation")
            mapping = conversation.get("mapping", {})
            
            # 寻找根节点 (parent 为 None 的节点)
            root_id = None
            for node_id, node in mapping.items():
                if node.get("parent") is None:
                    root_id = node_id
                    break
            
            if not root_id:
                continue
                
            # 遍历对话树 (简化处理：只取第一条分支)
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
                        # parts 可能包含非字符串内容，需过滤或转换
                        text_content = "".join([str(p) for p in parts if p is not None])
                    
                    if text_content and role in ["user", "assistant"]:
                        messages.append({"role": role, "content": text_content})
                
                # 移动到下一个节点
                children = node.get("children", [])
                if children:
                    current_id = children[0] # 默认走第一个分支
                else:
                    current_id = None
            
            # 将对话切分为 Q&A 对
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

