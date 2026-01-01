import torch
import os
import time
import logging
from typing import List, Dict, Any, Optional
from utils import ContentType, discover_models
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from constants import PROJECT_ROOT, MODEL_DIR, OFFLOAD_DIR

# Configure logging
logger = logging.getLogger(__name__)


def discover_multimodal_models():
    all_models = discover_models()
    multimodal_map = {}

    # Keywords for multimodal models
    multimodal_keywords = ["vl", "janus", "llava", "vision", "multimodal", "clip"]

    for name, path in all_models.items():
        name_lower = name.lower()
        if any(kw in name_lower for kw in multimodal_keywords):
            multimodal_map[name] = path

    return multimodal_map


models = discover_multimodal_models()


def load_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(image_path_or_url).convert("RGB")


class BaseMultimodalModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = models.get(model_name)
        if not self.model_path:
            self.model_path = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        raise NotImplementedError

    def chat(self, messages: List[Dict], **kwargs) -> str:
        raise NotImplementedError


class JanusModel(BaseMultimodalModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.vl_gpt = None

    def load_model(self):
        if self.vl_gpt is not None:
            return

        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
        except ImportError:
            raise ImportError(
                "DeepSeek-Janus package not found. \n"
                "Please install it from source: git clone https://github.com/deepseek-ai/Janus.git"
            )

        self.processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            self.model_path
        )
        self.tokenizer = self.processor.tokenizer

        is_mps = torch.backends.mps.is_available()
        is_cuda = torch.cuda.is_available()
        load_in_8bit = os.getenv("JANUS_LOAD_IN_8BIT", "true").lower() == "true"

        start_time = time.time()
        print(f"Loading Janus model: {self.model_name}...")

        is_mps = torch.backends.mps.is_available()
        is_cuda = torch.cuda.is_available()
        load_in_8bit = os.getenv("JANUS_LOAD_IN_8BIT", "true").lower() == "true"

        abs_offload_dir = os.path.join(PROJECT_ROOT, OFFLOAD_DIR)
        os.makedirs(abs_offload_dir, exist_ok=True)

        # Hardware and dtype configuration
        kwargs = {
            "low_cpu_mem_usage": True,
            "offload_folder": abs_offload_dir,
        }

        if is_mps:
            self.target_dtype = torch.float16
            kwargs.update(
                {
                    "torch_dtype": self.target_dtype,
                    "device_map": "auto",
                }
            )
        elif is_cuda:
            self.target_dtype = torch.float16 if load_in_8bit else torch.bfloat16
            kwargs.update(
                {
                    "torch_dtype": self.target_dtype,
                    "device_map": "auto",
                    "load_in_8bit": load_in_8bit,
                }
            )
        else:
            self.target_dtype = torch.float32
            kwargs.update(
                {
                    "torch_dtype": self.target_dtype,
                }
            )

        # Load model using the imported class
        self.vl_gpt = MultiModalityCausalLM.from_pretrained(self.model_path, **kwargs)

        if not is_mps and not is_cuda:
            self.vl_gpt = self.vl_gpt.to("cpu")

        self.vl_gpt.eval()
        print(f"Janus model loaded in {time.time() - start_time:.2f}s")

    def chat(self, messages: list[dict], **kwargs):
        self.load_model()

        try:
            from janus.utils.io import load_pil_images
        except ImportError:
            # Basic fallback if janus utils missing, though unlikely if package installed
            def load_pil_images(conv):
                return [
                    Image.open(img)
                    for c in conv
                    if "images" in c
                    for img in c["images"]
                ]

        conversation = []
        images = []

        for msg in messages:
            role = "<|User|>" if msg["role"] in ["user", "system"] else "<|Assistant|>"
            content = msg["content"]

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get("type") == ContentType.TEXT.value:
                        text_parts.append(part[ContentType.TEXT.value])
                    elif part.get("type") == ContentType.IMAGE_URL.value:
                        images.append(part[ContentType.IMAGE_URL.value])
                        text_parts.append("<image_placeholder>")

                conversation.append(
                    {
                        "role": role,
                        "content": "\n".join(text_parts),
                        "images": [img for img in images if img],
                    }
                )
            else:
                conversation.append({"role": role, "content": content})

        if conversation[-1]["role"] != "<|Assistant|>":
            conversation.append({"role": "<|Assistant|>", "content": ""})

        pil_images = load_pil_images(conversation)

        prepare_inputs = self.processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        if (
            hasattr(prepare_inputs, "pixel_values")
            and prepare_inputs.pixel_values is not None
        ):
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(
                self.target_dtype
            )

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        max_new_tokens = kwargs.get("max_new_tokens", 512)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=kwargs.get("do_sample", False),
            use_cache=True,
        )

        return self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )


class LlavaModel(BaseMultimodalModel):
    def __init__(self, model_name="llava-1.5-7b-hf"):
        super().__init__(model_name)

    def load_model(self):
        if self.model is not None:
            return

        from transformers import LlavaForConditionalGeneration, AutoProcessor

        print(f"Loading Llava model: {self.model_name}...")
        start_time = time.time()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path, low_cpu_mem_usage=True
            )

        print(f"Llava model loaded in {time.time() - start_time:.2f}s")

    def chat(self, messages: list[dict], **kwargs):
        self.load_model()

        # Simple prompt construction for Llava 1.5
        # It expects USER: <image>\n<text> ASSISTANT:

        prompt = ""
        images = []

        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]

            if role == "SYSTEM":
                # Llava 1.5 system prompt isn't always standard, often just prepended
                continue

            text_part = ""
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == ContentType.TEXT.value:
                        text_part += part[ContentType.TEXT.value]
                    elif part.get("type") == ContentType.IMAGE_URL.value:
                        images.append(load_image(part[ContentType.IMAGE_URL.value]))
                        text_part = "<image>\n" + text_part  # Image usually comes first
            else:
                text_part = content

            if role == "USER":
                prompt += f"USER: {text_part}\n"
            elif role == "ASSISTANT":
                prompt += f"ASSISTANT: {text_part}\n"

        if not prompt.strip().endswith("ASSISTANT:"):
            prompt += "ASSISTANT:"

        inputs = self.processor(
            text=prompt, images=images if images else None, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Determine max tokens
        max_new_tokens = kwargs.get("max_new_tokens", 200)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=kwargs.get("do_sample", True),
            temperature=kwargs.get("temperature", 0.7),
        )

        # Decode
        output = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract new text (remove prompt)
        # Check roughly where prompt ends or just return full and let caller handle,
        # but typically we want just the answer.
        # Llava generation includes input usually? No, generate() usually returns full sequence if not configured otherwise.

        # Use simple split for now
        if "ASSISTANT:" in output:
            return output.split("ASSISTANT:")[-1].strip()
        return output


class QwenVLModel(BaseMultimodalModel):
    def __init__(self, model_name="qwen3-vl-4b-instruct"):
        super().__init__(model_name)

    def load_model(self):
        if self.model is not None:
            return

        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )

        # Note: 'qwen_vl_utils' might be needed for advanced video/image processing

        print(f"Loading QwenVL model: {self.model_name}...")
        start_time = time.time()

        # Use Qwen2VL class as Qwen3VL is likely compatible or alias
        # If Qwen3 is not in standard transformers yet, we rely on AutoModel
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, dtype="auto", device_map="auto"
            )
        except Exception:
            # Fallback to generic auto model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
            )

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print(f"QwenVL model loaded in {time.time() - start_time:.2f}s")

    def chat(self, messages: list[dict], **kwargs):
        self.load_model()

        # QwenVL supports list of messages directly via processor.apply_chat_template if available
        # or we construct the input manually using qwen-vl format.
        # Qwen2-VL format:
        # [
        #   {'role': 'user', 'content': [{'type': 'image', 'image': '...'}, {'type': 'text', 'text': '...'}]}
        # ]

        qwen_messages = []
        for msg in messages:
            new_msg = {"role": msg["role"], "content": []}
            if isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == ContentType.TEXT.value:
                        new_msg["content"].append(
                            {"type": "text", "text": part[ContentType.TEXT.value]}
                        )
                    elif part.get("type") == ContentType.IMAGE_URL.value:
                        new_msg["content"].append(
                            {
                                "type": "image",
                                "image": part[ContentType.IMAGE_URL.value],
                            }
                        )
            else:
                new_msg["content"].append({"type": "text", "text": msg["content"]})
            qwen_messages.append(new_msg)

        # Prepare inputs
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=kwargs.get("max_new_tokens", 512)
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]


class MultimodalFactory:
    _models = {}

    @staticmethod
    def get_models():
        return list(models.keys())

    @classmethod
    def get_model(cls, model_name: str) -> BaseMultimodalModel:
        if model_name in cls._models:
            return cls._models[model_name]

        name_lower = model_name.lower()
        if "janus" in name_lower:
            instance = JanusModel(model_name)
        elif "llava" in name_lower:
            instance = LlavaModel(model_name)
        elif "vl" in name_lower:
            instance = QwenVLModel(model_name)
        else:
            logger.warning(
                f"Unknown multimodal model {model_name}, falling back to Janus"
            )
            instance = JanusModel(model_name)

        cls._models[model_name] = instance
        return instance
