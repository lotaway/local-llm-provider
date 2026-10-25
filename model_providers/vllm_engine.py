"""
vLLM 推理引擎实现
高性能推理引擎，支持 PagedAttention、连续批处理、Tensor Parallel
"""
import asyncio
import logging
import os
from typing import AsyncGenerator, Any, Optional, Dict

from .inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class VllmEngine(InferenceEngine):
    """
    vLLM 推理引擎封装
    
    优势：
    - PagedAttention 内存管理，显存利用率高
    - 连续批处理，吞吐率远超 Transformers
    - 原生 Tensor Parallel 多卡并行
    - 支持 AWQ/GPTQ/FP8 量化加速
    - Automatic Prefix Caching 复用系统提示词
    
    约束：
    - 仅支持 vLLM 注册的架构（config.json architectures 字段需匹配）
    - GGUF 支持实验性，不推荐生产使用
    - 强依赖 CUDA，Mac/CPU 支持有限
    - 自定义架构需编写 vLLM 侧实现并注册
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        quantization: Optional[str] = None,  # "awq", "gptq", "fp8", "marlin" 等
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        disable_custom_all_reduce: bool = False,
        dtype: str = "auto",  # "auto", "float16", "bfloat16", "float32"
        kv_cache_dtype: str = "auto",
        max_num_batched_tokens: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_log_stats: bool = True,
        **kwargs,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.dtype = dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.disable_log_stats = disable_log_stats
        self.extra_kwargs = kwargs

        self._engine = None
        self._tokenizer = None
        self._initialized = False
        
        # 延迟初始化，首次调用 generate_stream 时初始化
        # 这样可以在 UnifiedModelLoader 中先做兼容性检查

    def _init_engine(self):
        """延迟初始化 vLLM 引擎"""
        if self._initialized:
            return

        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "vLLM not installed. Install with: pip install vllm\n"
                f"Original error: {e}"
            )

        # 检查 CUDA 可用性
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("vLLM requires CUDA. No GPU available.")
            gpu_count = torch.cuda.device_count()
            if self.tensor_parallel_size > gpu_count:
                logger.warning(
                    f"tensor_parallel_size={self.tensor_parallel_size} > "
                    f"available GPUs={gpu_count}, adjusting to {gpu_count}"
                )
                self.tensor_parallel_size = gpu_count
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            trust_remote_code=self.trust_remote_code,
            enforce_eager=self.enforce_eager,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            dtype=self.dtype,
            kv_cache_dtype=self.kv_cache_dtype,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            disable_log_stats=self.disable_log_stats,
            **self.extra_kwargs,
        )

        logger.info(
            f"Initializing vLLM engine: model={self.model_path}, "
            f"tp={self.tensor_parallel_size}, "
            f"gpu_util={self.gpu_memory_utilization}, "
            f"max_len={self.max_model_len}, "
            f"quant={self.quantization}"
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 加载 tokenizer 用于格式化 prompt
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=self.trust_remote_code
        )
        
        self._initialized = True
        logger.info(f"vLLM engine loaded successfully: {self.model_path}")

    @property
    def model_type(self) -> str:
        return "vllm"

    @property
    def tokenizer(self):
        """获取 tokenizer，用于 chat template 格式化"""
        if not self._initialized:
            self._init_engine()
        return self._tokenizer

    async def generate_stream(
        self, 
        prompt: str, 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式生成
        
        Args:
            prompt: 输入提示词（已格式化的字符串）
            **kwargs: 生成参数
                - max_new_tokens: 最大生成 token 数
                - temperature: 采样温度
                - top_p: nucleus sampling 参数
                - top_k: top-k sampling 参数
                - stop: 停止词列表
                - repetition_penalty: 重复惩罚
                - presence_penalty: 存在惩罚
                - frequency_penalty: 频率惩罚
                - min_p: min-p sampling
                - seed: 随机种子
        """
        if not self._initialized:
            self._init_engine()

        from vllm import SamplingParams
        from vllm.utils import random_uuid

        # 构建采样参数
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", -1),
            stop=kwargs.get("stop"),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            presence_penalty=kwargs.get("presence_penalty", 0.0),
            frequency_penalty=kwargs.get("frequency_penalty", 0.0),
            min_p=kwargs.get("min_p", 0.0),
            seed=kwargs.get("seed"),
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )

        request_id = random_uuid()
        results_generator = self._engine.generate(prompt, sampling_params, request_id)

        prev_text = ""
        finish_reason = None
        
        try:
            async for request_output in results_generator:
                if not request_output.outputs:
                    continue
                    
                text = request_output.outputs[0].text
                delta = text[len(prev_text):]
                prev_text = text

                if delta:
                    yield delta

                # 检查是否结束
                if request_output.finished:
                    finish_reason = request_output.outputs[0].finish_reason
                    yield {"content": "", "finish_reason": finish_reason}
                    break

                await asyncio.sleep(0)  # 让出控制权

        except Exception as e:
            logger.error(f"vLLM generation error: {e}")
            raise

    def unload(self):
        """卸载模型，释放资源"""
        if self._engine is not None:
            # vLLM AsyncLLMEngine 没有显式 shutdown 方法
            # 依赖 Python GC 和进程退出清理 GPU 内存
            self._engine = None
            
        if self._tokenizer is not None:
            self._tokenizer = None
            
        self._initialized = False
        
        # 强制清理
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        logger.info(f"vLLM engine unloaded: {self.model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息，用于调试和监控"""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "quantization": self.quantization,
            "initialized": self._initialized,
        }