from unittest.mock import Mock, AsyncMock
from schemas.execution_protocol import (
    ExecutionRequest,
    ExecutionResponse,
    ExecutionStatus,
)
from adapters.http_adapter import HttpAdapter


class TestHttpAdapter:
    def test_execute_raises_not_implemented(self):
        adapter = HttpAdapter()
        request = ExecutionRequest(session_id="s1", user_message="hello")

        async def run():
            try:
                await adapter.execute(request)
                assert False
            except NotImplementedError:
                pass

        import asyncio

        asyncio.run(run())

    def test_handle_decision_raises_not_implemented(self):
        adapter = HttpAdapter()

        async def run():
            try:
                await adapter.handle_decision("s1", True, "ok")
                assert False
            except NotImplementedError:
                pass

        import asyncio

        asyncio.run(run())

    def test_get_status_returns_moved_message(self):
        adapter = HttpAdapter()

        async def run():
            status = await adapter.get_status("s1")
            assert status["status"] == "agent_execution_moved_to_client"

        import asyncio

        asyncio.run(run())
