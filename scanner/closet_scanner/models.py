"""온디바이스 모델 내려받기와 무결성 확인.

모델은 MediaPipe 공개 저장소의 TFLite 파일이다. 앱(iOS/Android)에서도 같은 파일을
쓸 수 있어, 여기서 검증한 결과가 기기 내 처리로 그대로 옮겨진다.
저장 위치는 CLOSET_MODELS_DIR 또는 scanner/.models (시스템 경로 비의존).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "segmenter": (
        f"{BASE}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
        "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0",
    ),
    "pose_task": (
        f"{BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
    ),
    "embedder": (
        f"{BASE}/image_embedder/mobilenet_v3_large/float32/latest/mobilenet_v3_large.tflite",
        "11af3c560dfeed7737cb4c03c23bf52a8403020784192d4dea0b74862a12828d",
    ),
}


def models_dir() -> Path:
    env = os.environ.get("CLOSET_MODELS_DIR")
    d = Path(env) if env else Path(__file__).resolve().parent.parent / ".models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(name: str) -> Path:
    url, digest = MODELS[name]
    path = models_dir() / url.rsplit("/", 1)[1]
    if path.exists() and _sha256(path) == digest:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"모델 내려받는 중: {path.name}")
    urllib.request.urlretrieve(url, tmp)
    got = _sha256(tmp)
    if got != digest:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{path.name} 해시 불일치: {got}")
    tmp.replace(path)
    return path


def model_path(name: str) -> Path:
    """name: segmenter | pose | embedder."""
    if name == "pose":
        task = _fetch("pose_task")
        out = models_dir() / "pose_landmarks_detector.tflite"
        if not out.exists():
            with zipfile.ZipFile(task) as z:
                out.write_bytes(z.read("pose_landmarks_detector.tflite"))
        return out
    return _fetch(name)


def ensure_all() -> None:
    for n in ("segmenter", "pose", "embedder"):
        model_path(n)
