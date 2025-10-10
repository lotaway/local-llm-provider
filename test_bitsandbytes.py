import torch
from importlib.metadata import PackageNotFoundError, version
from transformers import (
    BitsAndBytesConfig,
)


def test_bitsandbytes():
    try:
        # Ensure bitsandbytes is actually installed before constructing the config
        version("bitsandbytes")
    except PackageNotFoundError:
        print("bitsandbytes is not installed or not available on this platform. Skipping test.")
        return

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    print("Successfully created BitsAndBytesConfig:", quantization_config)

if __name__ == '__main__':
    test_bitsandbytes()