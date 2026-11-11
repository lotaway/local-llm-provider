# GPU base image is vendor-specific and selectable at build time:
# NVIDIA (default) --build-arg BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
# AMD             --build-arg BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}
WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && python scripts/check_torch_build.py --constraints /tmp/torch-constraints.txt \
    && pip install --upgrade-strategy only-if-needed -c /tmp/torch-constraints.txt ".[tts]" \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple
ENV PORT=11434
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
