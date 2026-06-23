from schemas.execution_protocol import ExecutionRequest, ExecutionResponse
from adapters.base_adapter import TransportAdapter


class HttpAdapter(TransportAdapter):
    async def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        raise NotImplementedError("Agent execution moved to client side")

    async def handle_decision(
        self, session_id: str, approved: bool, feedback: str, data: dict | None = None
    ) -> ExecutionResponse:
        raise NotImplementedError("Agent execution moved to client side")

    async def handle_client_result(
        self, session_id: str, result_data: dict, success: bool, error: str
    ) -> ExecutionResponse:
        raise NotImplementedError("Agent execution moved to client side")

    async def get_status(self, session_id: str) -> dict:
        return {"status": "agent_execution_moved_to_client"}
