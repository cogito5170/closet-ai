"""색 처리: sRGB→CIELAB, 대표색 추출, 기본 색명.

색명은 Berlin & Kay(1969)의 기본 색채어 수준으로 거칠게 둔다. 조명과
화이트밸런스 때문에 세밀한 색명은 사진마다 흔들리기 때문이다.
"""

from __future__ import annotations

import numpy as np

# 기본 색 원형 (sRGB). 이름은 (영문 키, 한국어)
_PROTOTYPES = {
    "black": ((25, 25, 28), "검정"),
    "white": ((242, 242, 238), "흰색"),
    "gray": ((128, 128, 128), "회색"),
    "lightgray": ((190, 190, 190), "밝은 회색"),
    "beige": ((215, 195, 160), "베이지"),
    "brown": ((110, 70, 40), "갈색"),
    "red": ((190, 30, 35), "빨강"),
    "pink": ((235, 150, 175), "분홍"),
    "orange": ((235, 120, 30), "주황"),
    "yellow": ((240, 210, 50), "노랑"),
    "khaki": ((120, 120, 70), "카키"),
    "green": ((40, 140, 70), "초록"),
    "mint": ((150, 210, 185), "민트"),
    "skyblue": ((140, 185, 230), "하늘색"),
    "blue": ((40, 90, 190), "파랑"),
    "denim": ((70, 100, 140), "데님"),
    "navy": ((25, 35, 70), "네이비"),
    "purple": ((110, 60, 150), "보라"),
}

COLOR_KO = {k: v[1] for k, v in _PROTOTYPES.items()}


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) uint8 또는 0~255 float → (..., 3) Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


_PROTO_KEYS = list(_PROTOTYPES)
_PROTO_LAB = srgb_to_lab(np.array([_PROTOTYPES[k][0] for k in _PROTO_KEYS]))


def lab_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """밝기 차이를 절반만 반영한 Lab 거리. 같은 옷도 조명에 따라 L이 크게 흔들린다."""
    d = np.asarray(x) - np.asarray(y)
    return np.sqrt((0.5 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def color_name(lab: np.ndarray) -> str:
    d = np.linalg.norm(_PROTO_LAB - np.asarray(lab)[None], axis=1)
    return _PROTO_KEYS[int(d.argmin())]


def dominant_colors(pixels_rgb: np.ndarray, k: int = 3, seed: int = 0) -> list[dict]:
    """마스크 안 픽셀(N,3)의 대표색을 k-means로 찾는다. 비율 큰 순서."""
    if len(pixels_rgb) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(pixels_rgb) > 4000:
        pixels_rgb = pixels_rgb[rng.choice(len(pixels_rgb), 4000, replace=False)]
    lab = srgb_to_lab(pixels_rgb)
    k = min(k, len(lab))
    # k-means++ 초기화
    centers = [lab[rng.integers(len(lab))]]
    for _ in range(1, k):
        d2 = np.min([np.sum((lab - c) ** 2, axis=1) for c in centers], axis=0)
        p = d2 / d2.sum() if d2.sum() > 0 else None
        centers.append(lab[rng.choice(len(lab), p=p)])
    centers = np.array(centers)
    for _ in range(20):
        assign = np.argmin(((lab[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([lab[assign == i].mean(0) if np.any(assign == i) else centers[i] for i in range(k)])
        if np.allclose(new, centers, atol=0.1):
            break
        centers = new
    fracs = np.bincount(assign, minlength=k) / len(lab)
    order = np.argsort(-fracs)
    out: list[dict] = []
    for i in order:
        if fracs[i] < 0.08:
            continue
        name = color_name(centers[i])
        same = next((c for c in out if c["name"] == name), None)
        if same:  # 같은 색명은 하나로 합친다 (주름·그림자 때문에 갈라진 군집)
            f0, f1 = same["frac"], float(fracs[i])
            same["lab"] = [round((a * f0 + b * f1) / (f0 + f1), 2) for a, b in zip(same["lab"], centers[i])]
            same["frac"] = round(f0 + f1, 3)
            continue
        out.append({"lab": [round(float(v), 2) for v in centers[i]], "frac": round(float(fracs[i]), 3),
                    "name": name})
    return out


def palette_distance(p1: list[dict], p2: list[dict]) -> float:
    """두 대표색 목록의 거리. 각 색을 상대 목록의 가장 가까운 색에 맞춰 비율 가중 평균(대칭)."""
    if not p1 or not p2:
        return 100.0

    def one_way(a, b):
        lb = np.array([c["lab"] for c in b])
        tot = 0.0
        for c in a:
            tot += c["frac"] * float(lab_distance(np.array(c["lab"])[None], lb).min())
        return tot / sum(c["frac"] for c in a)

    return 0.5 * (one_way(p1, p2) + one_way(p2, p1))
