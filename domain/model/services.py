"""
领域层：模型选择领域服务
封装模型选择、引擎匹配等核心业务逻辑
"""
from typing import List, Optional
from domain.model.entities import Model, ModelId, ModelPath, ModelType, EngineType
from domain.model.repositories import ModelRepository


class ModelSelectionService:
    """模型选择领域服务"""
    
    def __init__(self, repository: ModelRepository):
        self._repository = repository
    
    def list_available(self) -> List[Model]:
        """获取所有可用模型"""
        return self._repository.list_models()
    
    def get_model(self, model_id: ModelId) -> Optional[Model]:
        """获取指定模型"""
        return self._repository.get_model(str(model_id))
    
    def resolve_model(self, model_id: str, preferred_engine: EngineType = EngineType.AUTO) -> Model:
        """解析模型（核心业务逻辑：架构兼容性检查、引擎回退）"""
        model = self._repository.get_model(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model not found: {model_id}")
        
        # 如果指定了引擎，检查兼容性
        if preferred_engine != EngineType.AUTO:
            if not model.supports_engine(preferred_engine):
                raise EngineIncompatibleError(
                    f"Model {model_id} does not support engine {preferred_engine.value}"
                )
            model.engine_type = preferred_engine
        else:
            # Auto 模式：选择最佳引擎
            model.engine_type = model.preferred_engine()
        
        return model
    
    def find_by_path(self, path: ModelPath) -> Optional[Model]:
        """根据路径查找模型"""
        return self._repository.find_by_path(path.value)
    
    def refresh_catalog(self) -> None:
        """刷新模型目录"""
        self._repository.refresh()


class TokenBudgetService:
    """Token 预算领域服务"""
    
    def __init__(self, default_context_length: int = 8192, reserve_tokens: int = 1024):
        self._default_context_length = default_context_length
        self._reserve_tokens = reserve_tokens
    
    def calculate_budget(self, context_length: int, max_new_tokens: int) -> int:
        """计算可用于历史消息的 token 预算"""
        available = context_length - max_new_tokens - self._reserve_tokens
        return max(0, available)
    
    def truncate_messages(
        self, 
        messages: List[dict], 
        token_counter: callable,
        budget: int
    ) -> List[dict]:
        """截断消息以适应预算（保留 system 消息）"""
        if not messages:
            return []
        
        system_msg = None
        start_idx = 0
        if messages[0].get("role") == "system":
            system_msg = messages[0]
            start_idx = 1
        
        history = messages[start_idx:]
        
        # 从最旧的消息开始移除，直到符合预算
        while history:
            test_msgs = ([system_msg] if system_msg else []) + history
            total_tokens = sum(token_counter(msg) for msg in test_msgs)
            
            if total_tokens <= budget:
                break
            
            if len(history) <= 1:
                # 只剩一条历史消息，无法继续截断
                break
            
            history.pop(0)
        
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(history)
        return result


class ModelNotFoundError(Exception):
    """模型未找到异常"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class EngineIncompatibleError(Exception):
    """引擎不兼容异常"""
    def __init__(self, message: str):
        super().__init__(message)