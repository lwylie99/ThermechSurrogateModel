# ============================================================
# PyTorch + CUDA Dockerfile for Machine Learning Projects
# ============================================================
# Requirements:
#   - NVIDIA GPU with CUDA support
#   - NVIDIA Container Toolkit installed on the host:
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#
# Build:  docker build -t ml-project .
# Run:    docker run --gpus all -it --rm \
#           -v $(pwd):/workspace \
#           -p 8888:8888 \
#           ml-project
# ============================================================

# Base image: official PyTorch image with CUDA + cuDNN
# Change the tag to match your CUDA version:
#   pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime   (smaller, runtime only)
#   pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel      (larger, includes compilers)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# ---------- Labels ----------
LABEL maintainer="your-name"
LABEL description="PyTorch CUDA machine learning environment"

# ---------- Environment ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        vim \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working directory ----------
WORKDIR /workspace

# ---------- Python dependencies ----------
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------- Copy project files ----------
COPY . .

# ---------- Default command ----------
# Starts a bash shell; override with your training script as needed:
#   docker run ... ml-project python train.py
CMD ["/bin/bash"]
