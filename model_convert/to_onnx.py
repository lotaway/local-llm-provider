import torch
from transformers import AutoModel, AutoTokenizer
import argparse
import os

def convert_to_onnx(model_name, output_path):
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model.eval()
    
    # Create dummy input
    dummy_input = tokenizer("Hello world", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    attention_mask = dummy_input["attention_mask"]
    
    # Export
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=17
    )
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Embedding Model to ONNX")
    parser.add_argument("--model", required=True, help="HF Model ID or path")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    
    args = parser.parse_args()
    
    convert_to_onnx(args.model, args.output)
