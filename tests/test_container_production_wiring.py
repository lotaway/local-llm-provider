import ast
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def imported_names(relative_path):
    tree = ast.parse(source(relative_path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chat_routes_import_every_command_they_execute():
    names = imported_names("interfaces/http/routes/chat_routes.py")
    required = {
        "List", "ChatCommand", "CompletionCommand",
        "LoadModelCommand", "UnloadModelCommand", "ListModelsQuery",
    }
    assert required <= names


def test_chat_routes_module_compiles():
    compile(source("interfaces/http/routes/chat_routes.py"), "chat_routes.py", "exec")


def test_wire_routes_injects_the_rag_service():
    text = source("bootstrap/container.py")
    assert "self._wire_rag_routes()" in text
    assert "set_rag_service(self.rag_service())" in text


def test_container_shutdown_uses_port_keys():
    text = source("bootstrap/container.py")
    assert "from domain.scheduling.ports import BatchScheduler, TaskScheduler" in text
    assert "AsyncTaskScheduler in self._singletons" not in text


def test_local_rag_service_does_not_import_milvus_store():
    text = source("infrastructure/rag/local_rag_service.py")
    assert "from infrastructure.rag.milvus_store import MilvusVectorStore" not in text
    assert '__all__ = ["LocalRAGService"]' in text


def test_container_feature_flag_is_injectable():
    settings = load_module("llp_container_settings", "infrastructure/config/settings.py")
    assert settings.Settings().container_enabled is True
    assert settings.Settings(container_enabled=False).container_enabled is False


def test_lifespan_starts_and_stops_the_container():
    text = source("main.py")
    assert "await init_container_runtime()" in text
    assert "await shutdown_container()" in text


def test_application_includes_both_container_routers():
    text = source("main.py")
    assert "app.include_router(chat_router)" in text
    assert "app.include_router(rag_router)" in text


def test_rag_routes_expose_stable_endpoints():
    text = source("interfaces/http/routes/rag_routes.py")
    assert 'prefix="/v1/rag"' in text
    assert "def set_rag_service(" in text
    assert '@router.post("/documents")' in text
    assert '@router.post("/retrieve")' in text
    assert '@router.post("/generate")' in text
    assert '@router.get("/health")' in text


def test_new_rag_route_module_compiles():
    compile(source("interfaces/http/routes/rag_routes.py"), "rag_routes.py", "exec")
