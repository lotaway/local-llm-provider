"""
领域层：推理引擎端口（接口）
定义推理能力的抽象契约，基础设施层实现
"""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any, Optional, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceRequest:
    """推理请求值对象"""
    prompt: str
    config: "GenerationConfig"  # 前向引用
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})


@dataclass(frozen=True)
class InferenceChunk:
    """推理流式响应片段"""
    content: str = ""
    finish_reason: Optional[str] = None
    reasoning_content: Optional[str] = None
    token_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


# 延迟导入避免循环依赖
from domain.model.entities import GenerationConfig


class InferenceEngine(ABC):
    """推理引擎接口 - 领域层抽象"""
    
    @property
    @abstractmethod
    def engine_type(self) -> str:
        """引擎类型标识"""
        ...
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """当前加载的模型 ID"""
        ...
    
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """最大上下文长度"""
        ...
    
    @abstractmethod
    async def generate_stream(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[InferenceChunk, None]:
        """流式生成"""
        ...
    
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceChunk:
        """非流式生成（一次性返回完整结果）"""
        ...
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算 token 数量"""
        ...
    
    @abstractmethod
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """应用聊天模板"""
        ...
    
    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放资源"""
        ...
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        ...
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        ...


class ModelLoader(ABC):
    """模型加载器接口"""
    
    @abstractmethod
    def load(self, model_path: str, config: Dict[str, Any]) -> InferenceEngine:
        """加载模型并返回引擎实例"""
        ...
    
    @abstractmethod
    def can_load(self, model_path: str) -> bool:
        """检查是否能加载该路径的模型"""
        ...
    
    @abstractmethod
    def supported_types(self) -> list[str]:
        """支持的模型类型"""
        ...
    
    @abstractmethod
    def unload(self, engine: InferenceEngine) -> None:
        """卸载模型"""
        ...