import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOADER = ROOT / "utils" / "mcp_loader.py"
FACTORY = ROOT / "clients" / "agents" / "runtime_factory.py"
MAIN = ROOT / "main.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def test_loader_has_no_module_level_registry_instance():
    for node in _tree(LOADER).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assert not (isinstance(target, ast.Name) and target.id == "_connections")


def test_lifecycle_functions_require_an_explicit_registry():
    signatures = {
        node.name: [item.arg for item in node.args.args]
        for node in _tree(LOADER).body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("load_from_env", "get_connections", "stop_all", "_start_connection"):
        assert "registry" in signatures[name], name


def test_factory_registers_the_registry_on_the_runtime():
    source = FACTORY.read_text(encoding="utf-8")
    assert 'runtime.state.context["mcp_registry"] = registry' in source
    assert "load_from_env(mcp, registry)" in source


def test_shutdown_uses_the_runtime_registry():
    source = MAIN.read_text(encoding="utf-8")
    assert 'registry = context.get("mcp_registry")' in source
    assert "stop_all(registry)" in source
