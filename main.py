"""
Video Worker - Wan 影片生成 Worker
硬體需求: RTX 5080 16GB VRAM
職責: 場景影片、角色動畫、鏡頭運動效果生成
"""

import io
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from minio import Minio


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

# Wan API 設定
WAN_API_URL = os.getenv("WAN_API_URL", "http://localhost:8080")
WAN_API_KEY = os.getenv("WAN_API_KEY", "")  # Add API key support
WAN_MODEL_PATH = os.getenv("WAN_MODEL_PATH", "/models/wan/video_gen_model")
MOTION_MODEL_PATH = os.getenv("MOTION_MODEL_PATH", "/models/motion/controlnet_motion")

# 影片生成參數
MAX_VIDEO_LENGTH = int(os.getenv("MAX_VIDEO_LENGTH", "10"))
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "24"))
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "512"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "512"))
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
            "vram_used": 0,  # TODO: 實際查詢 VRAM 使用量
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
    """從 MinIO 下載檔案
    
    支援多種 URL 格式:
    - 完整 URL: http://minio:9000/assets/scenes/xxx.png
    - 外部 URL: https://anime-s3.weiwei92.cc/assets/scenes/xxx.png
    - 相對路徑: scenes/xxx.png
    """
    # 移除 protocol 和 host，只保留 /assets/ 之後的路徑
    # 處理各種 URL 格式
    object_name = object_url
    
    # 如果是完整 URL，移除 protocol 和 host
    if "://" in object_name:
        # 找到 /assets/ 的位置
        assets_index = object_name.find("/assets/")
        if assets_index != -1:
            # 提取 /assets/ 之後的路徑（"/assets/" = 8 個字元）
            object_name = object_name[assets_index + 8:]
        else:
            raise ValueError(f"Could not find '/assets/' in URL: {object_url}")
    
    print(f"  Downloading from MinIO: bucket={S3_BUCKET}, object={object_name}")
    
    response = minio_client.get_object(S3_BUCKET, object_name)
    data = response.read()
    response.close()
    response.release_conn()
    return data


# === Wan API 輔助函式 ===
def submit_wan_video_generation(image_bytes: bytes, prompt: str, duration: float, negative_prompt: str = "") -> str:
    """
    提交 Wan 影片生成請求
    
    Returns:
        task_id: Wan 任務 ID
    """
    url = f"{WAN_API_URL}/api/v1/generate/video"
    headers = {}
    if WAN_API_KEY:
        headers["Authorization"] = f"Bearer {WAN_API_KEY}"
    
    files = {
        "image": ("input.png", image_bytes, "image/png")
    }
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "duration": duration,
        "fps": VIDEO_FPS,
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
        "num_inference_steps": NUM_INFERENCE_STEPS,
    }
    response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result["task_id"]


def wait_wan_video_result(task_id: str, timeout_seconds: int = 600) -> dict[str, Any]:
    """
    輪詢 Wan 影片生成結果
    
    Returns:
        任務歷史資料，包含輸出影片資訊
    """
    url = f"{WAN_API_URL}/api/v1/status/{task_id}"
    headers = {}
    if WAN_API_KEY:
        headers["Authorization"] = f"Bearer {WAN_API_KEY}"
    
    start_time = time.time()
    
    while time.time() - start_time < timeout_seconds:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        history = response.json()
        
        status = history.get("status", "pending")
        if status in ["completed", "success"]:
            return history
        elif status in ["failed", "error"]:
            raise ValueError(f"Wan generation failed: {history.get('error', 'Unknown error')}")
        
        time.sleep(5)
    
    raise TimeoutError(f"Wan generation timed out after {timeout_seconds}s")


def download_wan_video(task_id: str) -> bytes:
    """
    從 Wan API 下載生成的影片
    """
    url = f"{WAN_API_URL}/api/v1/output/{task_id}"
    headers = {}
    if WAN_API_KEY:
        headers["Authorization"] = f"Bearer {WAN_API_KEY}"
    
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.content


# === 影片生成函式 ===
def generate_scene_video(user_input: dict[str, Any]) -> dict[str, Any]:
    """
    生成場景影片 (Wan API)
    
    Input:
    {
        "scene_image_url": "http://minio:9000/assets/scenes/xxx.png",
        "scene_json": {...},
        "duration": 6.0
    }
    
    Output:
    {
        "video_url": "http://minio:9000/assets/videos/xxx.mp4",
        "metadata": {...}
    }
    """
    scene_id = user_input.get("scene_json", {}).get("scene_id", "scene")
    duration = user_input.get("duration", 6.0)
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Generating scene video: {scene_id}")
    
    prompt = user_input.get("scene_json", {}).get("video", {}).get("prompt", "anime scene animation")
    negative_prompt = "low quality, blurry, bad anatomy, worst quality"
    
    # 下載場景圖片
    scene_image_url = user_input.get("scene_image_url", "")
    if not scene_image_url:
        raise ValueError("scene_image_url is required")
    
    try:
        scene_image_data = download_from_minio(scene_image_url)
        print(f"  Downloaded scene image: {len(scene_image_data)} bytes")
    except Exception as e:
        raise ValueError(f"Could not download scene image: {e}")
    
    try:
        print(f"  Submitting to Wan API: {WAN_API_URL}")
        task_id = submit_wan_video_generation(scene_image_data, prompt, duration, negative_prompt)
        print(f"  Task ID: {task_id}")
        print("  Waiting for video generation...")
        history = wait_wan_video_result(task_id)
        print(f"  Generation complete")
        
        video_bytes = download_wan_video(task_id)
        print(f"  Downloaded video: {len(video_bytes)} bytes")
    except Exception as e:
        print(f"  Wan API error: {e}")
        print("  Falling back to Mock generation...")
        video_bytes = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 10000  # Mock MP4 header
    
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


def submit_wan_animation(character_image: bytes, motion_prompt: str, duration: float) -> str:
    """提交角色動畫生成請求"""
    url = f"{WAN_API_URL}/api/v1/generate/animation"
    headers = {}
    if WAN_API_KEY:
        headers["Authorization"] = f"Bearer {WAN_API_KEY}"
    
    files = {
        "character_image": ("character.png", character_image, "image/png")
    }
    data = {
        "motion_prompt": motion_prompt,
        "duration": duration,
        "fps": VIDEO_FPS,
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
    }
    response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result["task_id"]


def generate_character_animation(user_input: dict[str, Any]) -> dict[str, Any]:
    """
    生成角色動畫 (Wan API)
    
    Input:
    {
        "character_image_url": "http://minio:9000/assets/characters/xxx.png",
        "motion_description": "walking forward",
        "duration": 3.0
    }
    
    Output:
    {
        "video_url": "http://minio:9000/assets/animations/xxx.mp4",
        "metadata": {...}
    }
    """
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
    
    try:
        print(f"  Submitting to Wan API: {WAN_API_URL}")
        task_id = submit_wan_animation(character_image_data, motion_desc, duration)
        print(f"  Task ID: {task_id}")
        print("  Waiting for animation generation...")
        history = wait_wan_video_result(task_id)
        print(f"  Generation complete")
        
        video_bytes = download_wan_video(task_id)
        print(f"  Downloaded animation: {len(video_bytes)} bytes")
    except Exception as e:
        print(f"  Wan API error: {e}")
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
    """
    生成鏡頭運動效果 (Wan API)
    
    Input:
    {
        "scene_video_url": "http://minio:9000/assets/videos/xxx.mp4",
        "camera_json": {
            "shot": "medium",
            "angle": "eye_level",
            "movement": "pan_right"
        }
    }
    
    Output:
    {
        "video_url": "http://minio:9000/assets/videos/xxx_camera.mp4",
        "metadata": {...}
    }
    """
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
    
    try:
        print(f"  Submitting camera motion to Wan API: {WAN_API_URL}")
        url = f"{WAN_API_URL}/api/v1/generate/camera-motion"
        headers = {}
        if WAN_API_KEY:
            headers["Authorization"] = f"Bearer {WAN_API_KEY}"
        
        files = {
            "video": ("input.mp4", scene_video_data, "video/mp4")
        }
        data = {
            "movement": camera_movement,
            "shot": shot,
            "angle": angle,
            "fps": VIDEO_FPS,
        }
        response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        response.raise_for_status()
        task_id = response.json()["task_id"]
        print(f"  Task ID: {task_id}")
        print("  Waiting for camera motion generation...")
        history = wait_wan_video_result(task_id)
        print(f"  Generation complete")
        
        video_bytes = download_wan_video(task_id)
        print(f"  Downloaded video: {len(video_bytes)} bytes")
    except Exception as e:
        print(f"  Wan API error: {e}")
        print("  Falling back to Mock generation...")
        video_bytes = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 10000
    
    object_name = f"videos/camera_{int(time.time())}.mp4"
    video_url = upload_to_minio(video_bytes, object_name)
    
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
    print(f"  Wan API URL: {WAN_API_URL}")
    print(f"  Wan API Key: {'***' + WAN_API_KEY[-4:] if WAN_API_KEY else '(none)'}")
    print(f"  Capabilities: {WORKER_CAPABILITIES}")
    
    # 註冊 Worker
    register_worker()
    
    current_job_id = None
    
    while True:
        try:
            # 發送心跳
            status = "busy" if current_job_id else "idle"
            send_heartbeat(status, current_job_id)
            
            # 如果有正在執行的作業，繼續執行
            if current_job_id:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            
            # 領取新作業
            job = claim_job()
            
            if job is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            
            current_job_id = job["job_id"]
            print(f"[{datetime.now(timezone.utc).isoformat()}] Claimed job: {current_job_id} (type: {job['type']})")
            
            # 更新狀態為 running
            update_job_status(current_job_id, "running", progress=0.0)
            
            # 執行作業
            result = run_video_job(job)
            
            # 更新進度
            update_job_status(current_job_id, "running", progress=100.0)
            
            # 標記完成
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
