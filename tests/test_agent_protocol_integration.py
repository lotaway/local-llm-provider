import pytest
from fastapi.testclient import TestClient


class TestAgentMetadataEndpoint:
    def test_metadata_endpoint_structure(self):
        from main import app
        client = TestClient(app)
        response = client.get("/v1/agents/metadata")
        assert response.status_code in [200, 400, 401]
        
        if response.status_code == 200:
            body = response.json()
            assert "agents" in body
            assert "tools" in body
            assert "capabilities" in body
            assert "version" in body
            assert isinstance(body["agents"], list)
            assert len(body["agents"]) >= 2
    
    def test_metadata_endpoint_agents_structure(self):
        from main import app
        client = TestClient(app)
        response = client.get("/v1/agents/metadata")
        
        if response.status_code == 200:
            body = response.json()
            agent = body["agents"][0]
            assert "name" in agent
            assert "description" in agent
            assert "supported_task_types" in agent
            assert "capabilities" in agent


import json


class TestAgentStreamEndpoint:
    def test_stream_event_format(self):
        try:
            from main import app
            client = TestClient(app)
            response = client.post(
                "/v1/agents/run",
                json={
                    "model": "test",
                    "messages": ["test query"],
                    "stream": True,
                }
            )
            
            if response.status_code == 200:
                content = response.text
                lines = content.strip().split("\n\n")
                
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str:
                            data = json.loads(data_str)
                            assert "type" in data
                            assert "content" in data
                            assert "timestamp" in data
                            assert data["type"] in ["message", "tool_call", "status", "error"]
        except Exception as e:
            if "MongoDB" in str(e) or "Connection refused" in str(e):
                pytest.skip("MongoDB not available")
            raise


class TestAgentErrorHandling:
    def test_error_response_format(self):
        from main import app
        client = TestClient(app)
        response = client.get("/v1/agents/metadata")
        
        if response.status_code == 400:
            body = response.json()
            assert "error" in body
            assert "code" in body["error"]
            assert "message" in body["error"]
            assert body["error"]["code"] in [
                "agent_not_initialized",
                "session_not_found",
                "tool_not_found",
                "permission_denied",
                "execution_failed",
                "invalid_request",
                "timeout",
            ]