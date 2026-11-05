import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = "services/tts_preference_store.py"
FACTORY = "infrastructure/tts/preference_collection.py"


def _imports(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_business_layer_does_not_import_pymongo():
    assert "pymongo" not in _imports(STORE)


def test_store_has_no_hardcoded_database_or_uri():
    source = (ROOT / STORE).read_text(encoding="utf-8")
    assert '"rag_system"' not in source
    assert "MONGO_URI" not in source
    assert "MongoClient" not in source


def test_unconfigured_store_raises_instead_of_building_a_client():
    source = (ROOT / STORE).read_text(encoding="utf-8")
    assert "raise TTSPreferenceStoreUnconfigured" in source


def test_collection_factory_reads_injectable_configuration():
    source = (ROOT / FACTORY).read_text(encoding="utf-8")
    assert "from pymongo import MongoClient" in source
    assert "MONGO_DB_NAME" in source
    assert "TTS_MONGO_COLLECTION_PROFILES" in source
    assert "def build_preference_collection(database: str = MONGO_DB_NAME)" in source


def test_composition_root_injects_the_collection():
    source = (ROOT / "bootstrap" / "tts_composition.py").read_text(encoding="utf-8")
    assert "TTSPreferenceStore(collection=build_preference_collection())" in source
