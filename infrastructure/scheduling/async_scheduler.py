"""
基础设施层：异步任务调度器实现
支持优先级队列、并发控制、批处理
"""
import asyncio
import logging
import heapq
import time
from typing import Any, Callable, Awaitable, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from domain.scheduling.ports import TaskScheduler, BatchScheduler, TaskResult

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedTask:
    """优先级任务"""
    priority: int
    task_id: str = field(compare=False)
    handler: Callable[[], Awaitable[Any]] = field(compare=False)
    created_at: float = field(compare=False)
    future: asyncio.Future = field(compare=False, default_factory=asyncio.Future)


class AsyncTaskScheduler(TaskScheduler):
    """异步任务调度器"""
    
    def __init__(self, max_concurrent: int = 10, max_queue_size: int = 1000):
        self._max_concurrent = max_concurrent
        self._max_queue_size = max_queue_size
        self._queue: list = []  # 优先级队列 (heapq)
        self._active: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, asyncio.Queue] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }
        self._lock = asyncio.Lock()
    
    async def start(self):
        """启动调度器"""
        if self._worker_task is None or self._worker_task.done():
            self._shutdown_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop())
    
    async def submit(
        self, 
        task_id: str, 
        handler: Callable[[], Awaitable[Any]],
        priority: int = 0
    ) -> asyncio.Queue:
        async with self._lock:
            if len(self._queue) >= self._max_queue_size:
                raise RuntimeError("Task queue is full")
            
            if task_id in self._active or task_id in self._results:
                raise ValueError(f"Task {task_id} already exists")
            
            result_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
            self._results[task_id] = result_queue
            
            future = asyncio.Future()
            task = PrioritizedTask(
                priority=-priority,  # 负数因为 heapq 是最小堆
                task_id=task_id,
                handler=handler,
                created_at=time.time(),
                future=future,
            )
            heapq.heappush(self._queue, task)
            self._stats["submitted"] += 1
            
            return result_queue
    
    async def cancel(self, task_id: str) -> bool:
        async with self._lock:
            # 从队列中移除
            for i, task in enumerate(self._queue):
                if task.task_id == task_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    if task_id in self._results:
                        del self._results[task_id]
                    self._stats["cancelled"] += 1
                    return True
            
            # 取消正在执行的任务
            if task_id in self._active:
                self._active[task_id].cancel()
                return True
        
        return False
    
    def get_queue_size(self) -> int:
        return len(self._queue)
    
    def get_active_count(self) -> int:
        return len(self._active)
    
    async def shutdown(self, graceful: bool = True) -> None:
        self._shutdown_event.set()
        
        if graceful:
            # 等待活跃任务完成
            while self._active:
                await asyncio.sleep(0.1)
        else:
            # 强制取消所有任务
            for task in self._active.values():
                task.cancel()
            await asyncio.gather(*self._active.values(), return_exceptions=True)
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "queue_size": len(self._queue),
            "active_count": len(self._active),
        }
    
    async def _worker_loop(self):
        """工作循环"""
        while not self._shutdown_event.is_set():
            async with self._lock:
                # 检查是否有空闲槽位
                if len(self._active) >= self._max_concurrent:
                    await asyncio.sleep(0.01)
                    continue
                
                if not self._queue:
                    await asyncio.sleep(0.01)
                    continue
                
                # 取出最高优先级任务
                task = heapq.heappop(self._queue)
                
                # 创建执行任务
                async def execute():
                    try:
                        result = await task.handler()
                        task.future.set_result(result)
                        self._stats["completed"] += 1
                    except asyncio.CancelledError:
                        task.future.cancel()
                        self._stats["cancelled"] += 1
                        raise
                    except Exception as e:
                        task.future.set_exception(e)
                        self._stats["failed"] += 1
                    finally:
                        async with self._lock:
                            self._active.pop(task.task_id, None)
                            if task.task_id in self._results:
                                queue = self._results.pop(task.task_id)
                                try:
                                    if not task.future.cancelled():
                                        if task.future.exception() is None:
                                            await queue.put(task.future.result())
                                        else:
                                            await queue.put(None)
                                except Exception:
                                    pass
                
                exec_task = asyncio.create_task(execute())
                self._active[task.task_id] = exec_task
            
            await asyncio.sleep(0)


class AsyncBatchScheduler(BatchScheduler):
    """异步批处理调度器 - 用于连续批处理"""
    
    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 10):
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._pending: Dict[str, asyncio.Queue] = {}
        self._batch_handler: Optional[Callable] = None
        self._processor_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    def set_batch_handler(self, handler: Callable) -> None:
        self._batch_handler = handler
    
    async def register(self, request_id: str, payload: Dict) -> asyncio.Queue:
        queue = asyncio.Queue()
        self._pending[request_id] = queue
        return queue
    
    async def unregister(self, request_id: str) -> None:
        self._pending.pop(request_id, None)
    
    async def process_batch(self, batch: list) -> Dict[str, Any]:
        if self._batch_handler is None:
            return {}
        return await self._batch_handler(batch)
    
    async def start(self):
        if self._processor_task is None or self._processor_task.done():
            self._shutdown = False
            self._processor_task = asyncio.create_task(self._process_loop())
    
    async def shutdown(self):
        self._shutdown = True
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
    
    async def _process_loop(self):
        while not self._shutdown:
            if not self._pending:
                await asyncio.sleep(0.001)
                continue
            
            # 收集批次
            batch_items = list(self._pending.items())[:self._max_batch_size]
            if not batch_items:
                await asyncio.sleep(0.001)
                continue
            
            batch = [{"request_id": rid, **payload} for rid, payload in batch_items]
            
            # 处理批次
            try:
                results = await self.process_batch(batch)
                for rid, result in results.items():
                    if rid in self._pending:
                        await self._pending[rid].put(result)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                for rid in [item[0] for item in batch_items]:
                    if rid in self._pending:
                        await self._pending[rid].put(None)
            
            await asyncio.sleep(self._max_wait_ms / 1000)