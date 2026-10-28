"""VLM 인스턴스 검출: 사진 한 장 → 옷 한 벌 단위 목록 (실험 경로, --detector vlm).

부위 분할(detect.py)은 몸을 상체·하체·발로 자르므로 원피스가 둘로 나뉘고 아우터와 이너가
한 영역으로 합쳐진다. 이 경로는 비전 언어 모델에게 옷 한 벌마다 종류와 bbox를 받아
제품의 등록 단위(인스턴스)와 맞춘다. 같은 옷 묶기는 기존 임베딩·색·질문 큐를 그대로 쓴다.

- 보내는 것은 긴 변 1024px로 줄인 사진 한 장이다. 결과는 사진 해시로 캐시한다.
- bbox는 모델의 시각 추정이며 confidence는 보정된 확률이 아니다.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .color import dominant_colors
from .detect import Garment, PhotoResult
from .imageio import load_image
from .tagger import ClaudeTagger

DEFAULT_MODEL = os.environ.get("CLOSET_DETECT_MODEL", "claude-sonnet-5-5")
SEND_SIDE = 1024

CATEGORIES = ["top", "bottom", "outer", "dress", "shoes", "other", "unknown"]
# 같은 옷 묶기에서 서로 비교할 수 있는 부위
REGION_OF = {"top": "upper", "outer": "upper", "dress": "upper", "bottom": "lower", "shoes": "shoes"}

DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "group", "no_person", "uncertain"]},
        "garments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "name_ko": {"type": "string"},
                    "color": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "number"}},
                    "confidence": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["category", "name_ko", "color", "bbox", "confidence", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["status", "garments"],
    "additionalProperties": False,
}

DETECT_PROMPT = """옷장 등록용으로 이 사진의 주 인물이 입은 옷을 한 벌씩 찾아라.

- status: 주 인물 한 명이면 ok, 비슷한 크기의 사람이 여럿이면 group, 사람이 없으면 no_person, 판별이 어려우면 uncertain.
- 옷 한 벌이 항목 하나다. 원피스는 상·하로 나누지 말고 dress 하나로 적는다.
- 겉옷 안의 상의가 보이면 outer와 top을 따로 적는다. 신발 한 켤레는 shoes 하나다.
- 가방·모자·벨트·액세서리·헤어밴드는 적지 않는다. 점프수트·케이프는 other.
- 가장자리에 걸친 다른 사람의 옷, 배경의 걸린 옷은 적지 않는다.
- bbox: [x0, y0, x1, y1], 사진 전체 기준 0~1. 보이는 부분만 감싼다. 신발은 두 발을 함께 감싼다.
- color: 한국어 색 이름(예: 검정, 아이보리, 연청). name_ko: 옷장에 보일 짧은 이름.
- confidence: 0~1, 종류와 위치 판단의 확신. note: 모호한 점이 있으면 한국어 한 문장, 없으면 빈 문자열.
- 가려져 판별할 수 없는 옷은 unknown으로 적고 추정 종류를 단정하지 않는다."""


class VLMDetector(ClaudeTagger):
    """ClaudeTagger의 호출·캐시·토큰 집계를 그대로 쓴다."""

    def __init__(self, cache_file: Path, model: str = DEFAULT_MODEL):
        super().__init__(cache_file, model=model)

    def detect(self, im: Image.Image) -> dict | None:
        small = im.copy()
        small.thumbnail((SEND_SIDE, SEND_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=90)
        data = buf.getvalue()
        key = "detect:" + self._key(data)
        if key not in self.cache:
            block = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                  "data": base64.standard_b64encode(data).decode("utf-8")}}
            out = self._ask([block, {"type": "text", "text": DETECT_PROMPT}], DETECT_SCHEMA)
            if out is None:
                return None
            self.cache[key] = out
            self.save()
        return self.cache[key]


def clean_bbox(b) -> list[float] | None:
    """모델이 준 bbox를 0~1 안의 [x0,y0,x1,y1]로 고친다. 넓이가 없으면 None."""
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in b)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        return None
    return [round(v, 4) for v in (x0, y0, x1, y1)]


def _square(im: Image.Image) -> Image.Image:
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def analyze_photo_vlm(path: Path, vis, det, photo_id: str, crops_dir: Path | None = None) -> PhotoResult:
    im = load_image(path)
    r = det.detect(im)
    if r is None:
        return PhotoResult(photo_id, "error", "검출 모델 응답 없음")
    if r["status"] in ("group", "no_person"):
        return PhotoResult(photo_id, r["status"], "검출 모델 판정")
    W, H = im.size
    rgb = np.asarray(im)
    items = [(d, clean_bbox(d.get("bbox"))) for d in r["garments"]]
    items = [(d, b) for d, b in items if b is not None]
    px = [(int(b[0] * W), int(b[1] * H), int(np.ceil(b[2] * W)), int(np.ceil(b[3] * H))) for _, b in items]
    res = PhotoResult(photo_id, "ok")
    for k, ((d, bbox), (x0, y0, x1, y1)) in enumerate(zip(items, px), 1):
        cat = d["category"]
        # 대표색은 상자 가운데 60%에서 뽑아 배경 섞임을 줄이고,
        # 상자 안에 든 다른 옷(아우터 속 상의 등)의 영역은 뺀다
        cw, ch = x1 - x0, y1 - y0
        keep = np.zeros(rgb.shape[:2], bool)
        keep[y0 + int(0.2 * ch):y1 - int(0.2 * ch), x0 + int(0.2 * cw):x1 - int(0.2 * cw)] = True
        own = keep.copy()
        for j, (a0, b0, a1, b1) in enumerate(px):
            if j != k - 1 and (a1 - a0) * (b1 - b0) < cw * ch:
                keep[b0:b1, a0:a1] = False
        if keep.sum() < 0.15 * own.sum():
            keep = own
        pix = rgb[keep] if keep.any() else rgb[y0:y1, x0:x1].reshape(-1, 3)
        gcrop = _square(im.crop((x0, y0, x1, y1)))
        flags = []
        if d["confidence"] < 0.5 or cat == "unknown":
            flags.append("low_confidence")
        g = Garment(
            id=f"{photo_id}#{k}", photo=photo_id, region=REGION_OF.get(cat, "other"),
            bbox=bbox, area_frac=round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 4),
            colors=dominant_colors(pix), flags=flags,
        )
        g.tags = {"category": cat, "name_ko": d["name_ko"], "color": d["color"],
                  "confidence": d["confidence"], "note": d["note"], "source": "vlm"}
        g.embedding = vis.embed(gcrop)
        if crops_dir is not None:
            fname = f"{photo_id.replace('/', '_')}__{k}.jpg"
            gcrop.save(crops_dir / fname, quality=88)
            g.crop_file = fname
        res.garments.append(g)
    if not res.garments:
        res.status, res.reason = "no_garment", "검출 모델이 옷을 찾지 못함"
    return res
