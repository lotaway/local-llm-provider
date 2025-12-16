import torch


class UnifiedModelLoader:
    def __init__(self, model_path: str, options: dict = None):
        """
        Unified Model Loader
        :param model_path: Path to the model file or directory
        :param options: Unified configuration dictionary.
               Support keys:
               - use_gpu (bool): Whether to use GPU. Default True.
               - context_length (int): Context window size. Default 4096.
               - quantization (str): None, '4bit', '8bit'. Default None.
               - device_map (str): For transformers, e.g., 'auto', 'cpu'.
               - n_gpu_layers (int): For GGUF.
               - gguf_kwargs (dict): Extra args for llama.cpp
               - onnx_kwargs (dict): Extra args for onnxruntime
               - transformers_kwargs (dict): Extra args for transformers
        """
        self.options = options or {}
        self.model_path = model_path

        # Unified parameters
        self.use_gpu = self.options.get("use_gpu", True)
        self.context_length = self.options.get("context_length", 4096)

        if model_path.endswith(".gguf"):
            self._load_gguf()
        elif model_path.endswith(".onnx"):
            self._load_onnx()
        else:
            self._load_transformers()

    def _load_gguf(self):
        from llama_cpp import Llama

        # Map unified params to GGUF params
        n_gpu_layers = self.options.get("n_gpu_layers", -1 if self.use_gpu else 0)

        gguf_kwargs = self.options.get("gguf_kwargs", {})

        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.context_length,
            n_gpu_layers=n_gpu_layers,
            verbose=self.options.get("verbose", False),
            **gguf_kwargs
        )
        self.model_type = "gguf"

    def _load_onnx(self):
        import onnxruntime as ort

        # Map unified params to ONNX params
        providers = self.options.get("providers")
        if not providers:
            if (
                self.use_gpu
                and "CUDAExecutionProvider" in ort.get_available_providers()
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        onnx_kwargs = self.options.get("onnx_kwargs", {})

        self.model = ort.InferenceSession(
            self.model_path, providers=providers, **onnx_kwargs
        )
        self.model_type = "onnx"

    def _load_transformers(self):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        transformers_kwargs = self.options.get("transformers_kwargs", {}).copy()

        # Handle quantization
        quantization = self.options.get("quantization")
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            transformers_kwargs["quantization_config"] = quantization_config
        elif quantization == "8bit":
            transformers_kwargs["load_in_8bit"] = True

        # Handle device map
        if self.use_gpu:
            transformers_kwargs.setdefault("device_map", "auto")
            transformers_kwargs.setdefault("torch_dtype", torch.float16)
        else:
            transformers_kwargs.setdefault("device_map", "cpu")

        # trust remote code is often needed
        transformers_kwargs.setdefault("trust_remote_code", True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **transformers_kwargs
        )
        self.model_type = "transformers"
