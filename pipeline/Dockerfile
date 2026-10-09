# GPU image for the reconstruction pipeline (cpipe + COLMAP + FFmpeg + gsplat).
#
# STATUS: written but NOT TESTED yet (no Linux + NVIDIA machine available).
# It mirrors the setup proven on Google Colab (torch 2.11 + CUDA 13.0 + gsplat
# 1.5.3). See docs/pipeline.md ("Docker") before relying on it.
#
# Build (from the repo root):
#   docker build -t colibrimo-pipeline --build-arg GIT_COMMIT=$(git rev-parse HEAD) pipeline
# Run:
#   docker run --gpus all --rm -v "$PWD/data:/work/data" colibrimo-pipeline \
#     run --video data/videos/room1.mp4 --output data/runs/room1 --preset balanced

FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ARG TORCH_VERSION=2.11.0
ARG GSPLAT_VERSION=1.5.3
# Pinned by gsplat v1.5.3 examples/requirements.txt
ARG FUSED_SSIM_COMMIT=328dc9836f513d00c4b5bc38fe30478b4435cbb5
# GPU architectures to compile for (no GPU is visible during `docker build`):
# 7.5 T4/RTX20, 8.0 A100, 8.6 RTX30/A10, 8.9 L4/RTX40, 9.0 H100
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ARG GIT_COMMIT=""

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    QT_QPA_PLATFORM=offscreen

# System tools: Python 3.12, COLMAP (CPU build) and FFmpeg from Ubuntu 24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev git build-essential ninja-build colmap ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# PyTorch built for CUDA 13.0 (matches the base image's nvcc)
RUN pip install "torch==${TORCH_VERSION}" torchvision --index-url https://download.pytorch.org/whl/cu130

# CUDA extensions compiled from source (no prebuilt wheels for this torch/CUDA)
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=4
RUN pip install --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@v${GSPLAT_VERSION}" \
    && pip install --no-build-isolation \
        "fused_ssim @ git+https://github.com/rahul-goel/fused-ssim@${FUSED_SSIM_COMMIT}"

# gsplat's training script and its Python requirements (numpy pin and the GPU
# packages installed above are left out, see docs/decisions.md D-005)
RUN git clone --depth 1 --branch "v${GSPLAT_VERSION}" \
        https://github.com/nerfstudio-project/gsplat.git /opt/gsplat \
    && grep -vE 'numpy|fused-ssim|fused-bilagrid|^#|^\s*$' \
        /opt/gsplat/examples/requirements.txt > /tmp/gsplat-requirements.txt \
    && pip install -r /tmp/gsplat-requirements.txt
ENV GSPLAT_EXAMPLES_DIR=/opt/gsplat/examples

# The Colibrimo pipeline
COPY . /opt/colibrimo-pipeline
RUN pip install /opt/colibrimo-pipeline
ENV CPIPE_GIT_COMMIT=${GIT_COMMIT}

WORKDIR /work
ENTRYPOINT ["cpipe"]
CMD ["--help"]
