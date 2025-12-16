import os
import subprocess
import argparse

def convert_to_gguf(model_path, output_path, quantization):
    """
    Wraps llama.cpp convert-hf-to-gguf.py
    Assumes llama.cpp is cloned at ../inference/llama/llama.cpp (hypothetically)
    or the user provides the path to the script.
    """
    
    # Check if we have the script
    llama_cpp_path = os.getenv("LLAMA_CPP_PATH", "./llama.cpp")
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    
    if not os.path.exists(convert_script):
        print(f"Error: {convert_script} not found. Please set LLAMA_CPP_PATH or clone llama.cpp")
        return

    cmd = [
        "python", convert_script,
        model_path,
        "--outfile", output_path,
        "--outtype", quantization
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SafeTensors to GGUF")
    parser.add_argument("--input", required=True, help="Path to HF model directory")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    parser.add_argument("--quant", default="q6_k", help="Quantization type (default: q6_k)")
    
    args = parser.parse_args()
    
    convert_to_gguf(args.input, args.output, args.quant)
