
from typing import List, Dict, Any
from langchain_core.documents import Document

class BaseChatLoader:
    def load(self, data: Any, source_file: str) -> List[Document]:
        raise NotImplementedError