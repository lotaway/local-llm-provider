# This script is for amd gpu used to install ROCm on Windows.
mamba run python3.12 & \
pip install --no-cache-dir \
https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm-0.1.dev0.tar.gz \
https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl \
https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_devel-0.1.dev0-py3-none-win_amd64.whl \
https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl \
https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torch-2.9.0%2Brocmsdk20251116-cp312-cp312-win_amd64.whl \
https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchaudio-2.9.0%2Brocmsdk20251116-cp312-cp312-win_amd64.whl \
https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/torchvision-0.24.0%2Brocmsdk20251116-cp312-cp312-win_amd64.whl