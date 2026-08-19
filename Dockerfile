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
# 2. 先安裝 torch（CUDA 12.4 版本，匹配基礎映像）
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 3. 安裝其他依賴（不包含 flash_attn）
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
    numpy>=1.24.0 \
    easydict \
    einops
# 4. 安裝 Wan 官方庫（不裝依賴，避免 flash_attn）
RUN pip install --no-cache-dir --no-deps git+https://github.com/Wan-Video/Wan2.1.git
# 5. Patch Wan 讓 flash_attn 變成可選（使用 PyTorch 內建 attention）
RUN python3 << 'PATCH_SCRIPT'
import os
f = '/opt/venv/lib/python3.10/site-packages/wan/modules/attention.py'
if os.path.exists(f):
    with open(f, 'r') as fh:
        content = fh.read()
    content = content.replace(
        'import flash_attn',
        'try:\n    import flash_attn\nexcept ImportError:\n    flash_attn = None'
    )
    with open(f, 'w') as fh:
        fh.write(content)
    print("Patched attention.py to make flash_attn optional")
else:
    print("attention.py not found, skipping patch")
PATCH_SCRIPT

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
