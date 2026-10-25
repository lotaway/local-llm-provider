"""
领域层：任务调度器端口
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable, Optional, Dict
from dataclasses import dataclass
import asyncio


@dataclass(frozen=True)
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class TaskScheduler(ABC):
    """任务调度器接口 - 用于并发控制、背压、批处理"""
    
    @abstractmethod
    async def submit(
        self, 
        task_id: str, 
        handler: Callable[[], Awaitable[Any]],
        priority: int = 0
    ) -> asyncio.Queue:
        """提交任务，返回结果队列"""
        ...
    
    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """取消任务"""
        ...
    
    @abstractmethod
    def get_queue_size(self) -> int:
        """获取队列大小"""
        ...
    
    @abstractmethod
    def get_active_count(self) -> int:
        """获取活跃任务数"""
        ...
    
    @abstractmethod
    async def shutdown(self, graceful: bool = True) -> None:
        """关闭调度器"""
        ...
    
    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        ...


class BatchScheduler(ABC):
    """批处理调度器接口 - 用于连续批处理优化"""
    
    @abstractmethod
    async def register(self, request_id: str, payload: Dict) -> asyncio.Queue:
        """注册请求"""
        ...
    
    @abstractmethod
    async def unregister(self, request_id: str) -> None:
        """注销请求"""
        ...
    
    @abstractmethod
    async def process_batch(self, batch: list) -> Dict[str, Any]:
        """处理批次（由实现类定时调用）"""
        ...
    
    @abstractmethod
    def set_batch_handler(self, handler: Callable) -> None:
        """设置批处理函数"""
        ...