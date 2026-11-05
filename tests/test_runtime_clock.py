import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_monotonic_clock_measures_elapsed_time_from_an_injected_source():
    module = load_module("llp_clock_elapsed", "infrastructure/runtime/monotonic_clock.py")
    readings = iter([100.0, 112.5])
    clock = module.MonotonicRuntimeClock(lambda: next(readings))
    assert clock.elapsed_seconds() == 12.5


def test_monotonic_clock_keeps_the_start_point_fixed():
    module = load_module("llp_clock_fixed", "infrastructure/runtime/monotonic_clock.py")
    readings = iter([5.0, 6.0, 9.0])
    clock = module.MonotonicRuntimeClock(lambda: next(readings))
    assert clock.elapsed_seconds() == 1.0
    assert clock.elapsed_seconds() == 4.0


def test_monotonic_clock_rejects_a_non_callable_time_source():
    module = load_module("llp_clock_guard", "infrastructure/runtime/monotonic_clock.py")
    try:
        module.MonotonicRuntimeClock(None)
    except TypeError:
        return
    raise AssertionError("non-callable time source was accepted")


def test_runtime_clock_port_is_an_abstract_contract():
    module = load_module("llp_clock_port", "domain/runtime/ports.py")
    assert "elapsed_seconds" in module.RuntimeClock.__abstractmethods__


def test_health_endpoint_reads_the_injected_clock():
    text = source("interfaces/http/routes/chat_routes.py")
    assert "def set_runtime_clock(" in text
    assert "uptime_seconds=_require_runtime_clock().elapsed_seconds()" in text
    assert "from domain.runtime.ports import RuntimeClock" in text


def test_container_wires_the_runtime_clock_implementation():
    text = source("bootstrap/container.py")
    assert "self._wire_runtime_clock()" in text
    assert "set_runtime_clock(self.runtime_clock())" in text
    assert "from infrastructure.runtime.monotonic_clock import MonotonicRuntimeClock" in text


def test_ci_targeted_suite_runs_the_clock_tests():
    text = source(".github/workflows/ci.yml")
    assert "tests/test_runtime_clock.py" in text
