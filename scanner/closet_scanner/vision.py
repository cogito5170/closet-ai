"""TFLite 모델 래퍼: 사람·의류 분할, 자세 랜드마크, 이미지 임베딩.

MediaPipe 태스크 런타임은 Linux에서 libEGL(시스템 라이브러리)을 요구하므로,
모델 파일만 받아 LiteRT 인터프리터로 직접 실행한다.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .models import model_path

# selfie_multiclass_256x256 클래스
BG, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)

# 자세 랜드마크 번호 (BlazePose 33점)
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


def _interpreter(path):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(path), num_threads=2)
    it.allocate_tensors()
    return it


def _run(it, x: np.ndarray) -> list[np.ndarray]:
    it.set_tensor(it.get_input_details()[0]["index"], x[None].astype(np.float32))
    it.invoke()
    return [it.get_tensor(d["index"]) for d in it.get_output_details()]


# 좌우 반전 시 서로 바뀌는 랜드마크 쌍
_FLIP = list(range(33))
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
               (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    _FLIP[_a], _FLIP[_b] = _b, _a


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


class Vision:
    """모델 셋을 한 번만 올려 두고 재사용한다."""

    def __init__(self):
        self._seg = _interpreter(model_path("segmenter"))
        self._pose = _interpreter(model_path("pose"))
        self._emb = _interpreter(model_path("embedder"))

    def segment(self, im: Image.Image, out_size: tuple[int, int] | None = None) -> np.ndarray:
        """클래스 맵(H,W) uint8. 256 해상도 확률을 out_size로 쌍선형 보간한 뒤 argmax."""
        # 좌우 반전 평균(TTA): 거울 셀카와 일반 사진에서 같은 결과가 나오게 한다
        x = np.asarray(im.resize((256, 256), Image.BILINEAR), np.float32) / 255.0
        logits = 0.5 * (_run(self._seg, x)[0][0] + _run(self._seg, x[:, ::-1].copy())[0][0][:, ::-1])
        w, h = out_size or im.size
        if (w, h) == (256, 256):
            return logits.argmax(-1).astype(np.uint8)
        up = np.stack([
            np.asarray(Image.fromarray(logits[..., c]).resize((w, h), Image.BILINEAR))
            for c in range(logits.shape[-1])
        ], axis=-1)
        return up.argmax(-1).astype(np.uint8)

    def pose(self, crop: Image.Image) -> tuple[float, np.ndarray]:
        """정사각 사람 crop → (존재 확률, (33,3) [x,y,visibility]) 좌표는 crop 픽셀."""
        w, h = crop.size
        x = np.asarray(crop.resize((256, 256), Image.BILINEAR), np.float32) / 255.0
        o1 = _run(self._pose, x)
        o2 = _run(self._pose, x[:, ::-1].copy())
        a = o1[0].reshape(-1, 5)[:33]
        b = o2[0].reshape(-1, 5)[:33][_FLIP]  # 반전 결과를 원래 좌우로 되돌림
        b = b.copy()
        b[:, 0] = 256.0 - b[:, 0]
        lm = 0.5 * (a + b)
        presence = 0.5 * (float(o1[1].ravel()[0]) + float(o2[1].ravel()[0]))
        pts = np.stack([lm[:, 0] * w / 256.0, lm[:, 1] * h / 256.0, _sigmoid(lm[:, 3])], axis=1)
        return presence, pts

    def embed(self, im: Image.Image) -> np.ndarray:
        """L2 정규화한 1280차원 임베딩. 거울 셀카를 위해 좌우 반전과 평균한다."""
        x = np.asarray(im.resize((224, 224), Image.BILINEAR), np.float32) / 255.0
        v = _run(self._emb, x)[0][0] + _run(self._emb, x[:, ::-1].copy())[0][0]
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)
