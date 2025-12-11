import torch
from transformers import AutoModelForCausalLM
import os
import time
from utils import ContentType


project_root = os.path.abspath(os.path.dirname(__file__))
models = {
    "deepseek-janus:7b": os.path.join(
        project_root, "..", "models", "deepseek-ai", "Janus-Pro-7B"
    )
}


class JanusModel:
    def __init__(self, model_name="deepseek-janus:7b"):
        self.model_path = models[model_name]
        self.vl_chat_processor = None
        self.tokenizer = None
        self.vl_gpt = None

    @staticmethod
    def get_models():
        return list(models.keys())

    def load_model(self):
        if self.vl_gpt is not None:
            return

        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
        except ImportError:
            MultiModalityCausalLM = None
            VLChatProcessor = None

        if MultiModalityCausalLM is None:
            raise ImportError(
                "DeepSeek-Janus package not found. \n"
                "Please install it from source: git clone https://github.com/deepseek-ai/Janus.git\n"
                "Note: 'pip install janus' installs an unrelated library."
            )
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.model_path
        )
        
        self.tokenizer = self.vl_chat_processor.tokenizer

        # Determine device and dtype for Mac M4 optimization
        is_mps = torch.backends.mps.is_available()
        is_cuda = torch.cuda.is_available()

        # Optimize for Mac M4 16GB: use 8-bit quantization to reduce memory
        load_in_8bit = os.getenv("JANUS_LOAD_IN_8BIT", "true").lower() == "true"

        start_time = time.time()
        print("Loading VLChatProcessor...")
        
        if is_mps:
            # Mac M4 with MPS
            # MPS doesn't support BFloat16 well, use Float16 instead
            print("Loading Janus model optimized for Mac M4 (MPS)...")
            self.target_dtype = torch.float16
            self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=False,
                dtype=self.target_dtype,
                low_cpu_mem_usage=True,
                device_map="auto",
                offload_folder="./offload",
            )
            self.vl_gpt.eval()

        elif is_cuda:
            # CUDA GPU
            print("Loading Janus model for CUDA...")
            if load_in_8bit:
                # Use 8-bit quantization for memory efficiency
                self.target_dtype = torch.float16  # 8-bit uses float16 for computation
                self.vl_gpt: MultiModalityCausalLM = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=False,
                        load_in_8bit=True,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                )
            else:
                self.target_dtype = torch.bfloat16
                self.vl_gpt: MultiModalityCausalLM = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=False,
                        dtype=self.target_dtype,
                        device_map="auto",
                        low_cpu_mem_usage=True,
                    )
                )
            self.vl_gpt.eval()

        else:
            # CPU fallback
            print("Loading Janus model for CPU (this may be slow)...")
            self.target_dtype = torch.float32
            self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=False,
                dtype=self.target_dtype,
                low_cpu_mem_usage=True,
            )
            self.vl_gpt = self.vl_gpt.to("cpu").eval()
            
        end_time = time.time()
        loading_time = end_time - start_time
        print(f"VLChatProcessor loaded in {loading_time:.2f} seconds")

    def chat(self, messages: list[dict], **kwargs):
        self.load_model()

        try:
            from janus.utils.io import load_pil_images
        except ImportError:
            load_pil_images = None

        # Convert standard messages to Janus format
        conversation = []
        images = []

        for msg in messages:
            if msg["role"] == "user" or msg["role"] == "system":
                role = "<|User|>"
            else:
                role = "<|Assistant|>"
            content = msg["content"]

            if isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for part in content:
                    if part.get("type") == ContentType.TEXT.value:
                        text_parts.append(part[ContentType.TEXT.value])
                    elif part.get("type") == ContentType.IMAGE_URL.value:
                        # Assuming image_url is a local path or we need to download it
                        # The snippet expects 'images' list in the dict to contain PIL images or paths
                        image_path = part[ContentType.IMAGE_URL.value]
                        images.append(image_path)
                        text_parts.append("<image_placeholder>")

                formatted_content = "\n".join(text_parts)
                conversation.append(
                    {
                        "role": role,
                        "content": formatted_content,
                        "images": [
                            img for img in images if img
                        ],  # Simplification: existing snippet maps 1 image per placeholder usually, but let's follow the structure
                    }
                )
            else:
                conversation.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        # Ensure last message is assistant for generation?
        # The snippet has {"role": "<|Assistant|>", "content": ""} at the end
        if conversation[-1]["role"] != "<|Assistant|>":
            conversation.append({"role": "<|Assistant|>", "content": ""})

        # load images and prepare for inputs
        # load_pil_images expects the conversation list structure
        pil_images = load_pil_images(conversation)

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        # Ensure all inputs use the same dtype as the model to avoid dtype mismatch
        # Convert input tensors to target dtype
        if (
            hasattr(prepare_inputs, "pixel_values")
            and prepare_inputs.pixel_values is not None
        ):
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(
                self.target_dtype
            )

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        # Reduce max_new_tokens default to save memory
        max_new_tokens = kwargs.get("max_new_tokens", 256)  # Reduced from 512

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=kwargs.get("do_sample", False),
            use_cache=True,
            # Memory optimization: reduce batch size if needed
            num_beams=kwargs.get("num_beams", 1),  # Beam search uses more memory
        )

        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        return answer
