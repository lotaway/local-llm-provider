#!/bin/bash
set -e

pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

BNB_PATH=$(python -c "import bitsandbytes; print(bitsandbytes.__path__[0])")
if [ ! -f "$BNB_PATH/libbitsandbytes_rocm72.so" ]; then
    echo "----------------------------------------------------------------"
    echo "bitsandbytes ROCm 7.2 binary not found. Compiling from source..."
    echo "----------------------------------------------------------------"
    
    if ! command -v cmake &> /dev/null; then
        apt-get update && apt-get install -y cmake
    fi

    TEMP_DIR="/tmp/bitsandbytes_build"
    rm -rf $TEMP_DIR
    git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git $TEMP_DIR
    cd $TEMP_DIR
    
    git fetch origin rocm-build-update:rocm-build-update || true
    git switch rocm-build-update || echo "Already on main or branch not found, continuing with main..."
    
    export BNB_ROCM_ARCH=1
    export ROCM_HOME=/opt/rocm
    export PATH=$ROCM_HOME/bin:$PATH
    export HIP_PATH=$ROCM_HOME
    
    cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx1100" .
    cmake --build .
    
    cp bitsandbytes/libbitsandbytes_rocm64.so "$BNB_PATH/libbitsandbytes_rocm72.so"
    
    echo "----------------------------------------------------------------"
    echo "Compilation successful! libbitsandbytes_rocm72.so is ready."
    echo "----------------------------------------------------------------"
    cd /app
fi

python main.py
