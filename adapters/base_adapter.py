from abc import ABC, abstractmethod
from schemas.execution_protocol import ExecutionRequest, ExecutionResponse


class TransportAdapter(ABC):
    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResponse:
        pass

    @abstractmethod
    async def handle_decision(
        self, session_id: str, approved: bool, feedback: str, data: dict | None = None
    ) -> ExecutionResponse:
        pass

    @abstractmethod
    async def handle_client_result(
        self, session_id: str, result_data: dict, success: bool, error: str
    ) -> ExecutionResponse:
        pass

    @abstractmethod
    async def get_status(self, session_id: str) -> dict:
        pass
