"""
领域层：模型仓储接口
定义模型发现和元数据查询的抽象契约
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass(frozen=True)
class ModelInfo:
    """模型元数据（值对象）"""
    model_id: str           # 唯一标识，如 "qwen2.5-7b-instruct"
    display_name: str       # 显示名称
    model_path: Path        # 本地路径或远程标识
    model_type: str         # "hf" | "gguf" | "onnx" | "awq" | "gptq"
    size_bytes: Optional[int] = None
    architecture: Optional[str] = None  # config.json 中的 architectures
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            object.__setattr__(self, 'tags', [])


class ModelRepository(ABC):
    """模型仓储接口 - 领域层抽象"""
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """列出所有可用模型"""
        ...
    
    @abstractmethod
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型详情"""
        ...
    
    @abstractmethod
    def find_by_path(self, path: Path) -> Optional[ModelInfo]:
        """根据路径查找模型"""
        ...
    
    @abstractmethod
    def exists(self, model_id: str) -> bool:
        """检查模型是否存在"""
        ...
    
    @abstractmethod
    def refresh(self) -> None:
        """刷新模型列表（重新扫描）"""
        ...