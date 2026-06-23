from schemas.execution_protocol import (
    ExecutionRequest,
    ExecutionResponse,
    ExecutionStatus,
)


class TestExecutionRequest:
    def test_create_request(self):
        request = ExecutionRequest(session_id="s1", user_message="hello")
        assert request.session_id == "s1"
        assert request.user_message == "hello"
        assert request.context is None
        assert request.stream is False
        assert request.start_agent == "qa"

    def test_request_with_context(self):
        request = ExecutionRequest(
            session_id="s1", user_message="hello", context={"key": "val"}, stream=True
        )
        assert request.context == {"key": "val"}
        assert request.stream is True


class TestExecutionResponse:
    def test_create_response(self):
        response = ExecutionResponse(status=ExecutionStatus.COMPLETED, content="done")
        assert response.status == ExecutionStatus.COMPLETED
        assert response.content == "done"

    def test_response_with_error(self):
        response = ExecutionResponse(
            status=ExecutionStatus.FAILED,
            content="",
            error_message="something went wrong",
        )
        assert response.status == ExecutionStatus.FAILED
        assert response.error_message == "something went wrong"


class TestExecutionStatus:
    def test_status_values(self):
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"
        assert ExecutionStatus.WAITING_HUMAN.value == "waiting_human"
        assert ExecutionStatus.WAITING_CLIENT.value == "waiting_client"
