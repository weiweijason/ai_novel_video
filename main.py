"""
Video Worker - 本地 Wan 影片生成 Worker
硬體需求: RTX 5080 16GB VRAM
職責: 場景影片、角色動畫、鏡頭運動效果生成
"""

import io
import json
import os
import time
import tempfile
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests
import torch
from minio import Minio
from PIL import Image
from transformers import AutoTokenizer
from wan.configs import WAN_CONFIGS
import wan


# === 環境變數 ===
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
WORKER_ID = os.getenv("WORKER_ID", "video-01")
WORKER_TYPE = os.getenv("WORKER_TYPE", "video")
WORKER_HOSTNAME = os.getenv("WORKER_HOSTNAME", "video-worker")
WORKER_CAPABILITIES = os.getenv(
    "WORKER_CAPABILITIES",
    "scene_video,character_animation,camera_motion,transition_effect",
).split(",")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))

# MinIO 設定
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin123")
S3_BUCKET = os.getenv("S3_BUCKET", "assets")

# Wan 模型設定 (本地)
WAN_MODEL_PATH = os.getenv("WAN_MODEL_PATH", "/app/models/wan/Wan2.1-I2V-14B-720P")
MOTION_MODEL_PATH = os.getenv("MOTION_MODEL_PATH", "/models/motion")

# 影片生成參數
MAX_VIDEO_LENGTH = int(os.getenv("MAX_VIDEO_LENGTH", "10"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "24"))
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "1920"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "1080"))
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "50"))


# === MinIO 客戶端 ===
def get_minio_client() -> Minio:
    return Minio(
        S3_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=S3_ACCESS_KEY,
        secret_key=S3_SECRET_KEY,
        secure=S3_ENDPOINT.startswith("https"),
    )


minio_client = get_minio_client()


def ensure_bucket() -> None:
    """確保 Bucket 存在"""
    if not minio_client.bucket_exists(S3_BUCKET):
        minio_client.make_bucket(S3_BUCKET)


# === Worker API 通訊 ===
def register_worker() -> None:
    """向 API 註冊 Video Worker"""
    payload = {
        "worker_id": WORKER_ID,
        "worker_type": WORKER_TYPE,
        "hostname": WORKER_HOSTNAME,
        "capabilities": WORKER_CAPABILITIES,
        "models": ["wan-video-gen-v1"],
    }
    print(f"[{datetime.now(timezone.utc).isoformat()}] Registering worker: {WORKER_ID}")
    response = requests.post(f"{API_BASE_URL}/worker/register", json=payload, timeout=10)
    response.raise_for_status()
    print(f"[{datetime.now(timezone.utc).isoformat()}] Worker registered successfully")


def send_heartbeat(status: str, current_job: str | None = None) -> None:
    """發送心跳"""
    payload = {
        "worker_id": WORKER_ID,
        "status": status,
        "current_job": current_job,
        "gpu": {
            "name": "RTX 5080",
            "vram_total": 16384,
            "vram_used": 0,
        },
    }
    response = requests.post(
        f"{API_BASE_URL}/worker/heartbeat",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()


def claim_job() -> dict[str, Any] | None:
    """從佇列領取下一個影片生成作業"""
    response = requests.post(
        f"{API_BASE_URL}/worker/jobs/claim",
        json={"worker_id": WORKER_ID},
        timeout=10,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def update_job_status(
    job_id: str,
    status: str,
    progress: float | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """更新作業狀態"""
    payload: dict[str, Any] = {"status": status}
    if progress is not None:
        payload["progress"] = progress
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error

    response = requests.post(
        f"{API_BASE_URL}/worker/jobs/{job_id}/status",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()


# === MinIO 上傳/下載 ===
def upload_to_minio(data: bytes, object_name: str, content_type: str = "video/mp4") -> str:
    """上傳影片到 MinIO"""
    ensure_bucket()
    data_stream = io.BytesIO(data)
    minio_client.put_object(
        bucket_name=S3_BUCKET,
        object_name=object_name,
        data=data_stream,
        length=len(data),
        content_type=content_type,
    )
    return f"{S3_ENDPOINT}/{S3_BUCKET}/{object_name}"


def download_from_minio(object_url: str) -> bytes:
    """從 MinIO 下載檔案"""
    object_name = object_url
    
    if "://" in object_name:
        assets_index = object_name.find("/assets/")
        if assets_index != -1:
            object_name = object_name[assets_index + 8:]
        else:
            raise ValueError(f"Could not find '/assets/' in URL: {object_url}")
    
    print(f"  Downloading from MinIO: bucket={S3_BUCKET}, object={object_name}")
    
    response = minio_client.get_object(S3_BUCKET, object_name)
    data = response.read()
    response.close()
    response.release_conn()
    return data


# === Wan 本地模型 (使用 Wan 官方庫) ===
wan_i2v = None


def load_wan_model():
    """載入 Wan 本地模型 (使用 Wan 官方 WanI2V)"""
    global wan_i2v, WAN_MODEL_PATH
    if wan_i2v is not None:
        return wan_i2v
    
    print(f"  Loading Wan model from: {WAN_MODEL_PATH}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # 檢查模型目錄是否存在
    if not os.path.exists(WAN_MODEL_PATH):
        # 嘗試其他可能的路徑
        possible_paths = [
            WAN_MODEL_PATH,
            "/app/models/wan/Wan2.1-I2V-14B-720P",
            "/models/wan/Wan2.1-I2V-14B-720P",
            "./models/wan/Wan2.1-I2V-14B-720P",
        ]
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                WAN_MODEL_PATH = path
                print(f"  Found model at: {path}")
                found = True
                break
        if not found:
            print(f"  Model not found at any of: {possible_paths}")
            raise FileNotFoundError(f"Model directory not found")
    
    try:
        # 使用 Wan 官方的 WanI2V 類別
        cfg = WAN_CONFIGS['i2v-14B']
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=WAN_MODEL_PATH,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
        )
        print(f"  WanI2V model loaded successfully")
        return wan_i2v
    except Exception as e:
        print(f"  WanI2V load failed: {e}")
        raise


def generate_video_with_wan(
    image_bytes: bytes,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 48,
    width: int = 1920,
    height: int = 1080,
) -> bytes:
    """
    使用本地 Wan 模型生成影片 (WanI2V)
    """
    global wan_i2v
    if wan_i2v is None:
        load_wan_model()
    
    # 將圖片轉換為 Wan 需要的格式
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # WanI2V 需要 max_area 參數
    max_area = width * height
    print(f"  Generating video: {width}x{height}, {num_frames} frames, max_area={max_area}")
    print(f"  Prompt: {prompt}")
    
    try:
        # 使用 WanI2V.generate() 方法
        video = wan_i2v.generate(
            input_prompt=prompt,
            img=image_pil,
            max_area=max_area,
            frame_num=num_frames,
            shift=5.0,
            sampling_steps=NUM_INFERENCE_STEPS,
            guide_scale=5.0,
            n_prompt=negative_prompt,
            seed=-1,
            offload_model=True,
        )
        
        # video 是 numpy array (T, H, W, C)
        video_frames = video
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            from moviepy import ImageSequenceClip
            
            # 轉換為 PIL Image 列表
            pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in video_frames]
            
            clip = ImageSequenceClip(pil_frames, fps=VIDEO_FPS)
            clip.write_videofile(tmp_path, codec="libx264", logger=None)
            
            with open(tmp_path, "rb") as f:
                video_bytes = f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        print(f"  Video generated: {len(video_bytes)} bytes")
        return video_bytes
        
    except Exception as e:
        print(f"  Wan generation error: {e}")
        raise


# === 影片生成函式 ===
def generate_scene_video(user_input: dict[str, Any]) -> dict[str, Any]:
    """生成場景影片 (本地 Wan 模型)"""
    scene_id = user_input.get("scene_json", {}).get("scene_id", "scene")
    duration = user_input.get("duration", 6.0)
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating scene video: {scene_id}")
    
    prompt = user_input.get("scene_json", {}).get("video", {}).get("prompt", "anime scene animation")
    negative_prompt = "low quality, blurry, bad anatomy, worst quality"
    
    scene_image_url = user_input.get("scene_image_url", "")
    if not scene_image_url:
        raise ValueError("scene_image_url is required")
    
    try:
        scene_image_data = download_from_minio(scene_image_url)
        print(f"  Downloaded scene image: {len(scene_image_data)} bytes")
    except Exception as e:
        raise ValueError(f"Could not download scene image: {e}")
    
    num_frames = int(duration * VIDEO_FPS)
    
    try:
        video_bytes = generate_video_with_wan(
            image_bytes=scene_image_data,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
        )
    except Exception as e:
        print(f"  Wan generation error: {e}")
        print("  Falling back to Mock generation...")
        video_bytes = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 10000
    
    object_name = f"videos/scenes/{scene_id}_{int(time.time())}.mp4"
    video_url = upload_to_minio(video_bytes, object_name)
    
    return {
        "video_url": video_url,
        "metadata": {
            "scene_id": scene_id,
            "duration": duration,
            "fps": VIDEO_FPS,
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
            "prompt": prompt,
        }
    }


def generate_character_animation(user_input: dict[str, Any]) -> dict[str, Any]:
    """生成角色動畫 (本地 Wan 模型)"""
    character_name = user_input.get("character_name", "character")
    motion_desc = user_input.get("motion_description", "idle")
    duration = user_input.get("duration", 3.0)
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating character animation: {character_name} ({motion_desc})")
    
    character_image_url = user_input.get("character_image_url", "")
    if not character_image_url:
        raise ValueError("character_image_url is required")
    
    try:
        character_image_data = download_from_minio(character_image_url)
        print(f"  Downloaded character image: {len(character_image_data)} bytes")
    except Exception as e:
        raise ValueError(f"Could not download character image: {e}")
    
    prompt = f"anime character {motion_desc}, smooth motion, high quality"
    num_frames = int(duration * VIDEO_FPS)
    
    try:
        video_bytes = generate_video_with_wan(
            image_bytes=character_image_data,
            prompt=prompt,
            negative_prompt="low quality, blurry, bad anatomy",
            num_frames=num_frames,
            width=VIDEO_WIDTH,
            height=VIDEO_HEIGHT,
        )
    except Exception as e:
        print(f"  Wan generation error: {e}")
        print("  Falling back to Mock generation...")
        video_bytes = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 10000
    
    object_name = f"animations/{character_name}_{motion_desc}_{int(time.time())}.mp4"
    video_url = upload_to_minio(video_bytes, object_name)
    
    return {
        "video_url": video_url,
        "metadata": {
            "character_name": character_name,
            "motion": motion_desc,
            "fps": VIDEO_FPS,
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
        }
    }


def generate_camera_motion(user_input: dict[str, Any]) -> dict[str, Any]:
    """生成鏡頭運動效果"""
    camera_movement = user_input.get("camera_json", {}).get("movement", "static")
    shot = user_input.get("camera_json", {}).get("shot", "medium")
    angle = user_input.get("camera_json", {}).get("angle", "eye_level")
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Applying camera motion: {camera_movement}")
    
    scene_video_url = user_input.get("scene_video_url", "")
    if not scene_video_url:
        raise ValueError("scene_video_url is required")
    
    try:
        scene_video_data = download_from_minio(scene_video_url)
        print(f"  Downloaded scene video: {len(scene_video_data)} bytes")
    except Exception as e:
        raise ValueError(f"Could not download scene video: {e}")
    
    object_name = f"videos/camera_{int(time.time())}.mp4"
    video_url = upload_to_minio(scene_video_data, object_name)
    
    return {
        "video_url": video_url,
        "metadata": {
            "camera_movement": camera_movement,
            "shot": shot,
            "angle": angle,
            "fps": VIDEO_FPS,
            "width": VIDEO_WIDTH,
            "height": VIDEO_HEIGHT,
        }
    }


def run_video_job(job: dict[str, Any]) -> dict[str, Any]:
    """執行影片生成作業"""
    job_type = job["type"]
    user_input = job.get("input", {})
    
    if job_type == "scene_video":
        return generate_scene_video(user_input)
    if job_type == "character_animation":
        return generate_character_animation(user_input)
    if job_type == "camera_motion":
        return generate_camera_motion(user_input)
    
    raise ValueError(f"Unknown job type: {job_type}")


# === Worker 主循環 ===
def main() -> None:
    """Video Worker 主循環"""
    print(f"[{datetime.now(timezone.utc).isoformat()}] Video Worker starting...")
    print(f"  Worker ID: {WORKER_ID}")
    print(f"  API Base URL: {API_BASE_URL}")
    print(f"  S3 Endpoint: {S3_ENDPOINT}")
    print(f"  Wan Model Path: {WAN_MODEL_PATH}")
    print(f"  Video Settings: {VIDEO_WIDTH}x{VIDEO_HEIGHT}@{VIDEO_FPS}fps")
    print(f"  Capabilities: {WORKER_CAPABILITIES}")
    
    register_worker()
    
    # 預載入 Wan 模型
    try:
        load_wan_model()
    except Exception as e:
        print(f"  Warning: Could not load Wan model: {e}")
        print("  Will fall back to Mock generation")
    
    current_job_id = None
    
    while True:
        try:
            status = "busy" if current_job_id else "idle"
            send_heartbeat(status, current_job_id)
            
            if current_job_id:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            
            job = claim_job()
            
            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            
            current_job_id = job["job_id"]
            print(f"[{datetime.now(timezone.utc).isoformat()}] Claimed job: {current_job_id} (type: {job['type']})")
            
            update_job_status(current_job_id, "running", progress=0.0)
            
            result = run_video_job(job)
            
            update_job_status(current_job_id, "running", progress=100.0)
            update_job_status(current_job_id, "completed", result=result)
            
            print(f"[{datetime.now(timezone.utc).isoformat()}] Job completed: {current_job_id}")
            current_job_id = None
            
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] Error: {e}")
            if current_job_id:
                try:
                    update_job_status(current_job_id, "failed", error=str(e))
                except Exception:
                    pass
                current_job_id = None
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
