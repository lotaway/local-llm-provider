#!/bin/bash
set -e

pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

WANTED_BIN=$(python -c "import bitsandbytes" 2>&1 | grep "Configured ROCm binary not found at" | sed 's/.*found at //')

if [ -z "$WANTED_BIN" ]; then
    BNB_PATH=$(python -c "import bitsandbytes; print(bitsandbytes.__path__[0])")
    ROCM_VER=$(cat /opt/rocm/.info/version | cut -d'.' -f1,2 | tr -d '.')
    WANTED_BIN="$BNB_PATH/libbitsandbytes_rocm${ROCM_VER}.so"
fi

if [ ! -f "$WANTED_BIN" ]; then
    echo "----------------------------------------------------------------"
    echo "bitsandbytes ROCm binary not found. Expected: $WANTED_BIN"
    echo "Compiling from source to match your environment..."
    echo "----------------------------------------------------------------"
    
    if ! command -v cmake &> /dev/null; then
        apt-get update && apt-get install -y cmake
    fi

    TEMP_DIR="/tmp/bitsandbytes_build"
    rm -rf $TEMP_DIR
    git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git $TEMP_DIR
    cd $TEMP_DIR
    
    export BNB_ROCM_ARCH="gfx90a;gfx942;gfx1100;gfx1101;gfx1150;gfx1151;gfx1200;gfx1201"
    export ROCM_HOME=/opt/rocm
    export PATH=$ROCM_HOME/bin:$PATH
    export HIP_PATH=$ROCM_HOME
    
    cmake -DCOMPUTE_BACKEND=hip \
          -DCMAKE_BUILD_TYPE=MinSizeRel \
          -DCMAKE_HIP_FLAGS="--offload-compress" \
          -DBNB_ROCM_ARCH="$BNB_ROCM_ARCH" .
    cmake --build .
    
    cp bitsandbytes/libbitsandbytes_rocm*.so "$WANTED_BIN"
    
    echo "----------------------------------------------------------------"
    echo "Compilation successful! Binary placed at: $WANTED_BIN"
    echo "----------------------------------------------------------------"
    cd /app
fi

python main.py
