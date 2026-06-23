import uuid
from unittest.mock import Mock, AsyncMock, patch
from acp.schema import (
    TextContentBlock,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
)
from adapters.protocol_bridge import (
    ProtocolBridge,
    SessionStore,
    ContentHandler,
    PromptResponder,
)


class TestSessionStore:
    def test_create_session(self):
        store = SessionStore()
        sid = store.create(cwd="/tmp")
        assert sid is not None
        assert store.has(sid)

    def test_has_nonexistent_session(self):
        store = SessionStore()
        assert store.has("nonexistent") is False

    def test_remove_session(self):
        store = SessionStore()
        sid = store.create(cwd="/tmp")
        assert store.remove(sid) is True
        assert store.has(sid) is False

    def test_list_sessions(self):
        store = SessionStore()
        s1 = store.create(cwd="/a")
        s2 = store.create(cwd="/b")
        sessions_a = store.list_sessions(cwd="/a")
        assert len(sessions_a) == 1
        assert sessions_a[0].session_id == s1
        all_sessions = store.list_sessions(cwd=None)
        assert len(all_sessions) == 2


class TestContentHandler:
    def test_extract_text_from_text_blocks(self):
        blocks = [
            TextContentBlock(type="text", text="hello"),
            TextContentBlock(type="text", text="world"),
        ]
        result = ContentHandler.extract_text(blocks)
        assert result == "hello\nworld"

    def test_extract_text_empty(self):
        result = ContentHandler.extract_text([])
        assert result == ""

    def test_extract_text_ignores_non_text(self):
        class NonTextBlock:
            type = "image"

        blocks = [TextContentBlock(type="text", text="hello"), NonTextBlock()]
        result = ContentHandler.extract_text(blocks)
        assert result == "hello"


class TestProtocolBridge:
    def test_initialize(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)
        result = None

        async def run():
            nonlocal result
            result = await bridge.initialize(protocol_version=1)

        import asyncio

        asyncio.run(run())
        assert result.protocol_version == 1

    def test_new_session(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)
        result = None

        async def run():
            nonlocal result
            result = await bridge.new_session(cwd="/tmp")

        import asyncio

        asyncio.run(run())
        assert result.session_id is not None

    def test_new_and_close_session(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)

        async def run():
            new_resp = await bridge.new_session(cwd="/tmp")
            sid = new_resp.session_id
            await bridge.close_session(session_id=sid)
            sessions_resp = await bridge.list_sessions()
            assert len(sessions_resp.sessions) == 0

        import asyncio

        asyncio.run(run())

    def test_prompt_with_no_text_returns_end_turn(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)

        async def run():
            resp = await bridge.prompt(prompt=[], session_id="any")
            assert resp.stop_reason == "end_turn"

        import asyncio

        asyncio.run(run())

    def test_list_sessions(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)

        async def run():
            await bridge.new_session(cwd="/a")
            await bridge.new_session(cwd="/b")
            resp = await bridge.list_sessions()
            assert len(resp.sessions) == 2

        import asyncio

        asyncio.run(run())

    def test_on_connect_sets_connection(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)
        conn = AsyncMock()
        bridge.on_connect(conn)
        assert bridge._connection is conn

    def test_authenticate_returns_default(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)

        async def run():
            resp = await bridge.authenticate(method_id="token")
            assert resp is not None

        import asyncio

        asyncio.run(run())

    def test_ext_method_returns_empty(self):
        llm = Mock()
        bridge = ProtocolBridge(llm)

        async def run():
            resp = await bridge.ext_method("test", {})
            assert resp == {}

        import asyncio

        asyncio.run(run())
