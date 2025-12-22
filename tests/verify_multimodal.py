import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from model_providers.multimodal_provider import (
        MultimodalFactory,
        JanusModel,
        LlavaModel,
        QwenVLModel,
    )

    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def test_factory():
    print("Testing Factory...")

    # Test Janus
    m1 = MultimodalFactory.get_model("deepseek-janus:7b")
    assert isinstance(m1, JanusModel), f"Expected JanusModel, got {type(m1)}"
    print("Janus model factory check passed.")

    # Test Llava
    m2 = MultimodalFactory.get_model("llava-1.5-7b-hf")
    assert isinstance(m2, LlavaModel), f"Expected LlavaModel, got {type(m2)}"
    print("Llava model factory check passed.")

    # Test Qwen
    m3 = MultimodalFactory.get_model("qwen3-vl-4b-instruct")
    assert isinstance(m3, QwenVLModel), f"Expected QwenVLModel, got {type(m3)}"
    print("Qwen model factory check passed.")

    # Test fallback
    m4 = MultimodalFactory.get_model("unknown-model")
    assert isinstance(
        m4, JanusModel
    ), f"Expected fallback to JanusModel, got {type(m4)}"
    print("Fallback check passed.")


if __name__ == "__main__":
    test_factory()
