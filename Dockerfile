FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/CLIPGCN

# System packages required by PyTorch, OpenCV, MediaPipe, YOLOv5, and webcam GUI display.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    libasound2 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgtk-3-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    portaudio19-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# The realtime object branch expects YOLOv5 at /workspace/yolov5 by default.
# X3D and CTR-GCN are project-specific external repositories; mount or copy them
# at /workspace/X3D and /workspace/CTR-GCN together with their checkpoints.
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /workspace/yolov5

COPY . /workspace/CLIPGCN

CMD ["python", "webcam_realtime.py", "--help"]
