# Local LLM Provider

This is use for [Meta Note](https://github.com/lotaway/meta-note.git) Client, as a local LLM to provide code completion and chat, agent(edit,apply) ability.

## Directory Structure

```
local-llm-provider
├── agents                     Agents: planning, routing, verification, error handling
│   └── task_agents            Task-specific agents
├── controllers                FastAPI controllers (LLM/RAG/files)
├── file_loaders               External data loaders (ChatGPT, DeepSeek)
├── globals                    Package init and global config
├── model_convert              Model conversion scripts (GGUF/ONNX)
├── model_providers            Model providers & inference engines (transformers, llama.cpp, ComfyUI, POE)
├── repositories               Data access (MongoDB, Neo4j)
├── retrievers                 RAG retrievers (ES, hybrid, graph, reranker)
├── routers                    API routers & versioning
├── schemas                    Structured data models (graphs, etc.)
├── scripts                    Init scripts / SQL
├── services                   Services (graph extraction)
├── skills                     MCP skills and registry
├── tests                      Test suites and verification samples
├── utils                      Utilities (scheduler, content types, cancellation)
├── models                     Local models (optional)
├── main.py                    Server entry point
├── rag.py                     RAG pipeline: indexing/search
├── auth.py                    Auth helpers
├── permission_manager.py      Permission management
├── constants.py               Constants
├── manifest.json              MCP/provider manifest
├── Dockerfile                 Docker build
├── docker-compose.env.yml     Docker Compose env
├── docker-compose-elasticsearch-init.sh  Elasticsearch init script
├── .env.example               Environment variables example
├── start.sh / build.sh / deploy.sh  Start/build/deploy scripts
├── HARDWARE.md / RAG_MIGRATION.md / LICENSE  Docs
```

## RAG Architecture

The system uses **MongoDB as Source of Truth** with unified storage for documents and chunks.

```
                    ┌─────────────┐
                    │   MongoDB   │
                    │ (事实源)    │
                    │ documents   │
                    │ chunks      │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Milvus    │ │Elasticsearch│ │   Neo4j     │
    │ chunk向量   │ │  全文索引   │ │ 结构索引    │
    │ chunk文本   │ │  chunk文本  │ │ (仅存ID)    │
    └─────────────┘ └─────────────┘ └─────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           │
                    通过 chunk_id / doc_id 关联
```

### Data Flow
1. Load documents from `DATA_PATH` (default: `./docs`)
2. Split into chunks and save to MongoDB
3. Generate embeddings → Milvus
4. Index full-text → Elasticsearch
5. Extract graph → Neo4j (entities & relations)

### Supported Formats
`.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.json` (incl. ChatGPT/DeepSeek), code files (`.py`, `.java`, etc.)

## Usage

### Download Models

Download the openai standard model from [Hugging Face](https://huggingface.co) and put it into the `model` directory.

Change the `model_name` in `model_provider.py` to the downloaded model name.

Run the `main.py` file to start the server.

Install Vscode Continue Plugin and connect to the server.

### Install Requirements

```bash
uv pip install -e .
```

## Voice ASR (VibeVoice) Integration

This project integrates VibeVoice‑ASR via environment variables. You need a local checkout of the official repository and point the paths accordingly.

- Environment variables
  - `VIBEVOICE_DIR`: absolute path to VibeVoice repo (e.g. `/opt/VibeVoice`)
  - `VIBEVOICE_MODEL`: ASR model id (default `microsoft/VibeVoice-ASR`)
  - `VIBEVOICE_SCRIPT`: path to `demo/vibevoice_asr_inference_from_file.py` under `VIBEVOICE_DIR`

Example `.env`:

```env
VIBEVOICE_DIR=/opt/VibeVoice
VIBEVOICE_MODEL=microsoft/VibeVoice-ASR
VIBEVOICE_SCRIPT=/opt/VibeVoice/demo/vibevoice_asr_inference_from_file.py
```

Install VibeVoice:

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

API endpoints (versioned under `/v1`):
- `POST /v1/voice/to/text`
  - Upload whole audio: form-data `audio=@file`
  - Streaming session: form-data `session_id=abc`, `stream=true` (SSE subscription)
  - Chunk upload: form-data `session_id=abc`, `chunk=@part.raw`
- `POST /v1/voice/chat/completions`
  - Upload short audio (<1min): form-data `audio=@file`, optional `model`, `stream`
  - Transcribes with VibeVoice then forwards to chat completions

## AMD ROCm / WSL Setup

### Overview

This project runs under WSL2 with AMD GPUs via the [ROCDXG (librocdxg)](https://github.com/ROCm/librocdxg) solution — a user-mode library that bridges the Linux ROCm runtime and the Windows GPU driver stack through Microsoft's DXCore interface (`/dev/dxg`). It replaces the legacy roc4wsl packaging model and is loosely coupled from both ROCm releases and Windows driver versions.

### Initial Environment Setup

#### 1. Windows Side

1. Install the compatible AMD Adrenalin driver from [AMD Drivers](https://www.amd.com/en/support/download/drivers.html) (minimum version recommended by the ROCDXG compatibility matrix).
2. Install WSL2 (Ubuntu 24.04 or 22.04). See [Microsoft WSL docs](https://learn.microsoft.com/en-us/windows/wsl/install).
3. Install the [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/) (needed for librocdxg build).

#### 2. WSL Side — Install ROCm + librocdxg

```bash
# Install ROCm packages (follow official quick-start)
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html

# Build and install librocdxg
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg

export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/'
mkdir -p build && cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make
sudo make install
cd ../..

# Verify GPU detection
rocminfo
# Expected: Agent with Name: gfx1100 (RX 7900 XTX)
```

#### 3. WSL Side — Environment Variables

Add these to `~/.bashrc`:

```bash
export ROCM_HOME=/opt/rocm
export HIP_PATH=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:$LD_LIBRARY_PATH
export HSA_ENABLE_DXG_DETECTION=1   # Required for ROCDXG
export HSA_ENABLE_SDMA=0
```

#### 4. Create Conda Environment & Install PyTorch

```bash
mamba create -n ai python=3.12.12
mamba activate ai

# Install PyTorch ROCm version (match your ROCm version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
```

### Updating Environment

Updating the Windows AMD driver may change the bundled ROCm version (e.g., 7.2 → 7.4). This breaks the dependency chain:

```
Windows Driver → ROCm Userspace Libs (/opt/rocm) → librocdxg → PyTorch ROCm → bitsandbytes
```

Each layer is compiled/linked against a specific ROCm version. A mismatch at any layer breaks GPU compute.

**1. Check the new bundled ROCm version**

If skipped: `rocminfo` fails to detect the GPU — the userspace libs (`libhip`, `librocblas`, etc.) mismatch the kernel driver.

**2. Update ROCm packages in WSL**

```bash
sudo apt update && sudo apt upgrade rocm-*
```

Or reinstall following the [ROCm Installation Quick Start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html).

If skipped: Same as step 1 — GPU undetectable.

**3. Rebuild librocdxg**

```bash
cd librocdxg && git pull
mkdir -p build && cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make && sudo make install
```

If skipped: HSA runtime can't find the DXG device → `torch.cuda.is_available()` returns `False`.

**4. Update PyTorch ROCm**

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm{version}
```

If skipped: PyTorch is linked against old HIP libraries; loading GPU kernels causes segfaults or `undefined symbol` errors.

**5. Rebuild bitsandbytes**

See "About bitsandbytes → Build for Linux ROCm" below.

If skipped: The old `libbitsandbytes_rocm{old}.so` looks for `libhipblas.so.2` but the new ROCm ships `libhipblas.so.3` → `OSError: cannot open shared object file` (the error you hit before).

### About bitsandbytes

AMD ROCm requires building bitsandbytes from source to match the exact ROCm version.

#### Build for Linux ROCm

```bash
mamba activate ai
cd /home/wayluk/bitsandbytes

# Set ROCm environment
export ROCM_HOME=/opt/rocm
export HIP_PATH=$ROCM_HOME
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:$LD_LIBRARY_PATH

# Install build dependencies
pip install scikit-build-core setuptools wheel build

# Build and install
CMAKE_ARGS="-DCOMPUTE_BACKEND=hip" pip install . --no-build-isolation
```

After a successful build you'll see:
```
[100%] Linking CXX shared library bitsandbytes/libbitsandbytes_rocm{version}.so
[100%] Built target bitsandbytes
```

The compiled `.so` and all Python modules are automatically installed to the current conda environment's `site-packages`.

> **Note**: `pip install . --no-build-isolation` reuses the current environment's build deps. If you skip `--no-build-isolation`, pip will temporarily install isolated build deps which also works but takes slightly longer.

#### Known Build Issue — hsa-runtime64

If cmake fails with:
```
CMake Error: The imported target "hsa-runtime64::hsa-runtime64" references the file
  "/opt/rocm/lib/libhsa-runtime64.so.1.15.60400" but this file does not exist.
```

Fix with a symlink:
```bash
cd /opt/rocm/lib/
ln -s libhsa-runtime64.so.1.14 libhsa-runtime64.so.1.15.60400
```

See [ROCm#3606](https://github.com/ROCm/ROCm/issues/3606) for details.

### Alternative: Windows without WSL

- **ZLuda**: Install CUDA PyTorch (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`). Note: ZLuda does not support bitsandbytes.
- **DirectML**: `pip install torch-directml`
- **Native Windows ROCm**: After ROCm 7.0, AMD provides native Windows ROCm. See [ROCm on Windows](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html).

## For Multiple Mac (Apple silicon) devices

You can use [exo](https://github.com/exo-explore/exo) to start the mac device as node, multiple devices as clusters, and let data flow between them to use both gpu.

# Problem

if have problem with:

```bash
from .CrossEncoder import CrossEncoder
File "{user}\miniforge3\envs\python3.12\Lib\site-packages\sentence_transformers\cross_encoder\CrossEncoder.py", line 1, in <module>
    from transformers import (
ImportError: cannot import name 'PreTrainedModel' from 'transformers'
```

Try to update/lowversion transformers and sentence_transformers, or use hack in file `sentence_transformers\cross_encoder\CrossEncoder.py` before `from transformers import [xxx]`

```python
import transformers
if not hasattr(transformers, "PreTrainedModel"):
    from transformers.modeling_utils import PreTrainedModel
    transformers.PreTrainedModel = PreTrainedModel

if not hasattr(transformers, "PreTrainedTokenizer"):
    from transformers.tokenization_utils import PreTrainedTokenizer
    transformers.PreTrainedTokenizer = PreTrainedTokenizer
```

