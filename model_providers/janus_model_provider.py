import torch
from transformers import AutoModelForCausalLM
import os

try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
except ImportError:
    MultiModalityCausalLM = None
    VLChatProcessor = None
    load_pil_images = None


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

    def load_model(self):
        if self.vl_gpt is not None:
            return

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

        # Determine device and dtype
        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16  # MPS supports float16, bfloat16 support varies
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        else:
            device = "cpu"
            dtype = torch.float32

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(dtype).to(device).eval()

    def chat(self, messages: list[dict], **kwargs):
        self.load_model()

        # Convert standard messages to Janus format
        conversation = []
        images = []

        for msg in messages:
            role = "<|User|>" if msg["role"] == "user" else "<|Assistant|>"
            content = msg["content"]

            if isinstance(content, list):
                # Handle multimodal content
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        # Assuming image_url is a local path or we need to download it
                        # The snippet expects 'images' list in the dict to contain PIL images or paths
                        image_path = part["image_url"]
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

        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            do_sample=kwargs.get("do_sample", False),
            use_cache=True,
        )

        answer = self.tokenizer.decode(
            outputs[0].cpu().tolist(), skip_special_tokens=True
        )
        return answer
