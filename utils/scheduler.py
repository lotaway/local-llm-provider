import asyncio
from typing import Callable, Awaitable, cast
import os
from collections import deque


class Scheduler[_T]:
    def __init__(
        self,
        handler: Callable[[list], Awaitable[dict]],
        max_batch=int(os.getenv("MAX_BATCH", 4)),
    ):
        self.handler = handler
        self.max_batch = max_batch
        self.waiting = cast(deque[tuple[int, _T, asyncio.Queue[_T]]], deque())
        self.running = {}
        self.lock = asyncio.Lock()

    async def register(self, payload):
        q = asyncio.Queue()
        async with self.lock:
            rid = id(q)
            self.waiting.append((rid, payload, q))
            self.running[rid] = q
        return rid, q

    async def quit(self, rid: int):
        async with self.lock:
            for i in self.waiting:
                if i[0] == rid:
                    self.waiting.remove(i)
                    self.running.pop(rid, None)
                    break

    async def loop(self):
        while True:
            await asyncio.sleep(0)
            batch = []
            async with self.lock:
                while self.waiting and len(batch) < self.max_batch:
                    batch.append(self.waiting.popleft())
            if not batch:
                await asyncio.sleep(0.01)
                continue
            out = await self.handler(batch)

            requeue_list = []
            for item in batch:
                rid, payload, q = item
                token = out.get(rid)
                if rid not in self.running:
                    continue
                await q.put(token)
                if token is None:
                    async with self.lock:
                        self.running.pop(rid, None)
                else:
                    requeue_list.append(item)

            if requeue_list:
                async with self.lock:
                    self.waiting.extend(requeue_list)
