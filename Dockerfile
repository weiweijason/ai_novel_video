# Video Worker - Wan 影片生成
# 硬體需求: RTX 5080 16GB VRAM

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 建立虛擬環境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安裝 Python 依賴
COPY requirements.txt .
# 1. 升級 pip 工具鏈
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging
# 2. 先安裝 torch（flash_attn 編譯需要）
RUN pip install --no-cache-dir torch torchvision
# 3. 安裝 flash_attn 編譯依賴 + flash_attn 本身
RUN pip install --no-cache-dir psutil \
    && pip install --no-cache-dir --no-build-isolation flash_attn
# 4. 安裝其他依賴（不包含 Wan 庫，因為它依賴 flash_attn）
RUN pip install --no-cache-dir \
    requests>=2.31.0 \
    minio>=7.2.0 \
    transformers>=4.37.0 \
    accelerate>=0.26.0 \
    diffusers>=0.27.0 \
    open-clip-torch \
    sentencepiece \
    decord>=0.6.0 \
    imageio>=2.33.0 \
    imageio-ffmpeg>=0.4.9 \
    moviepy>=1.0.3 \
    tqdm \
    numpy>=1.24.0
# 5. 最後安裝 Wan 官方庫（依賴 flash_attn）
RUN pip install --no-cache-dir git+https://github.com/Wan-Video/Wan2.1.git

# 複製程式碼
COPY . .

# 模型目錄 (掛載 Volume)
RUN mkdir -p /models/wan /models/motion

# 環境變數預設值
ENV API_BASE_URL=http://api:8000
ENV WORKER_ID=video-01
ENV WORKER_TYPE=video
ENV WORKER_HOSTNAME=video-worker
ENV S3_ENDPOINT=http://minio:9000
ENV S3_BUCKET=assets

# 禁用 Python 輸出緩衝，讓日誌即時顯示
ENV PYTHONUNBUFFERED=1

# 啟動命令 (使用 -u 參數確保輸出即時顯示)
CMD ["python", "-u", "main.py"]
