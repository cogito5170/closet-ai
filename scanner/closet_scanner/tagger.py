"""Claude 비전으로 옷 태깅과 애매한 쌍 검증 (선택).

- 보내는 것은 옷 영역만 남기고 얼굴·피부·배경을 회색으로 지운 crop이다(원본 업로드 최소화).
- 분석은 옷 묶음(item)당 1회, 결과는 crop 해시로 캐시한다(PLAN §9 "등록 시 1회").
- ANTHROPIC_API_KEY(또는 SDK가 읽는 다른 자격 증명)가 있어야 동작한다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

DEFAULT_MODEL = os.environ.get("CLOSET_TAG_MODEL", "claude-haiku-5-5")

CATEGORIES = ["top", "bottom", "outer", "dress", "shoes", "bag", "hat", "accessory"]

TAG_SCHEMA = {
    "type": "object",
    "properties": {
        "is_garment": {"type": "boolean"},
        "category": {"type": "string", "enum": CATEGORIES},
        "layered_inner_visible": {"type": "boolean"},
        "color": {"type": "string"},
        "pattern": {"type": "string", "enum": ["solid", "stripe", "check", "dot", "floral", "graphic", "logo", "other"]},
        "material": {"type": "string"},
        "fit": {"type": "string", "enum": ["slim", "regular", "loose", "oversized", "unknown"]},
        "season": {"type": "array", "items": {"type": "string", "enum": ["spring", "summer", "fall", "winter"]}},
        "mood": {"type": "array", "items": {"type": "string"}},
        "formality": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "name_ko": {"type": "string"},
        "distinctive": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["is_garment", "category", "layered_inner_visible", "color", "pattern", "material", "fit",
                 "season", "mood", "formality", "name_ko", "distinctive", "confidence"],
    "additionalProperties": False,
}

PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["same", "different", "unsure"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

TAG_PROMPT = """이 이미지는 사람 사진에서 옷 영역만 남기고 나머지를 회색으로 지운 것이다.
스캐너는 이 영역을 '{region}' 부위({region_ko})로 잘랐다. 보이는 옷 한 벌을 태깅하라.

- is_garment: 실제 옷·신발이면 true, 분할 오류(배경, 가구, 동물 등)면 false.
- category: {categories} 중 하나. 겉옷 아래 안쪽 옷이 보이면 겉옷을 고르고 layered_inner_visible=true.
- color: 한국어 기본 색명 하나 (예: 검정, 흰색, 네이비, 베이지).
- material: 한국어 추정 소재 (예: 면, 데님, 니트, 가죽). 모르면 "모름".
- mood: 분위기 형용 1~3개, 한국어.
- formality: 1(매우 캐주얼)~5(정장).
- name_ko: 옷장에 보일 짧은 이름 (예: "네이비 크루넥 니트").
- distinctive: 같은 옷을 다른 사진에서 알아보는 데 쓸 특징 (로고, 단추, 패턴, 디테일). 없으면 "없음".
- confidence: 0~1, 카테고리 판단의 확신."""

PAIR_PROMPT = """두 이미지는 같은 사람의 서로 다른 사진에서 옷 영역만 잘라낸 것이다(나머지는 회색).
두 영역이 '같은 옷 한 벌'인지 판단하라. 비슷한 다른 옷(같은 색의 다른 티셔츠 등)일 수 있다.
로고·단추·패턴·봉제선·실루엣 같은 개체 수준 단서를 근거로 삼고, 단서가 부족하면 unsure로 답하라.
reason은 한국어 한 문장."""

REGION_KO = {"upper": "상체", "lower": "하체", "shoes": "발"}


class ClaudeTagger:
    def __init__(self, cache_file: Path, model: str = DEFAULT_MODEL):
        try:
            import anthropic
        except ImportError as e:
            raise SystemExit("Claude 태깅에는 anthropic 패키지가 필요합니다: uv sync --extra claude") from e
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.cache_file = cache_file
        self.cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _image_block(self, path: Path) -> dict:
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}

    def _ask(self, content: list[dict], schema: dict) -> dict | None:
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}],
            )
        except self.anthropic.AuthenticationError as e:
            raise SystemExit("Claude 인증 실패: ANTHROPIC_API_KEY를 확인하세요.") from e
        except self.anthropic.RateLimitError as e:
            print(f"  요청 한도 초과, 건너뜀: {e}")
            return None
        except self.anthropic.APIStatusError as e:
            print(f"  API 오류 {e.status_code}, 건너뜀: {e.message}")
            return None
        except self.anthropic.APIConnectionError as e:
            print(f"  연결 오류, 건너뜀: {e}")
            return None
        self.calls += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return None
        text = next((b.text for b in resp.content if b.type == "text"), None)
        return json.loads(text) if text else None

    def _key(self, *parts: bytes) -> str:
        h = hashlib.sha256(self.model.encode())
        for p in parts:
            h.update(p)
        return h.hexdigest()[:24]

    def tag(self, crop: Path, region: str) -> dict | None:
        key = "tag:" + self._key(crop.read_bytes(), region.encode())
        if key not in self.cache:
            prompt = TAG_PROMPT.format(region=region, region_ko=REGION_KO.get(region, region),
                                       categories=", ".join(CATEGORIES))
            out = self._ask([self._image_block(crop), {"type": "text", "text": prompt}], TAG_SCHEMA)
            if out is None:
                return None
            self.cache[key] = out
            self.save()
        return self.cache[key]

    def same_item(self, a: Path, b: Path) -> dict | None:
        ka, kb = sorted([a.read_bytes(), b.read_bytes()])
        key = "pair:" + self._key(ka, kb)
        if key not in self.cache:
            out = self._ask([self._image_block(a), self._image_block(b), {"type": "text", "text": PAIR_PROMPT}],
                            PAIR_SCHEMA)
            if out is None:
                return None
            self.cache[key] = out
            self.save()
        return self.cache[key]

    def save(self):
        self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1))
