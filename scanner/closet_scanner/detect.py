"""사진 한 장 → 옷 영역(상의·하의·신발) 목록.

순서: 전체 분할로 사람 찾기 → 사람 crop 재분할 + 자세 → 자세 기준선으로 부위 분리
→ 부위별 마스크 crop, 대표색, 임베딩. 알고리즘 설명은 docs/SCANNER.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import vision as V
from .color import dominant_colors, lab_distance, srgb_to_lab

REGIONS = ("upper", "lower", "shoes")
NEUTRAL = (128, 128, 128)


@dataclass
class Config:
    work_side: int = 768          # 전체 분할 해상도
    crop_side: int = 512          # 사람 crop 해상도
    min_person_frac: float = 0.04  # 사람으로 볼 최소 면적(사진 대비)
    group_ratio: float = 0.35     # 둘째 사람이 첫째의 이 비율 이상이면 단체 사진
    min_face_frac: float = 0.002  # 얼굴로 셀 최소 면적(사진 대비)
    min_presence: float = 0.5     # 자세 모델 사람 존재 확률
    min_vis: float = 0.5          # 랜드마크 가시성
    min_region_frac: float = 0.012  # 부위 최소 면적(사람 crop 대비)
    include_group: bool = False   # 단체 사진에서도 가장 큰 사람을 쓸지


@dataclass
class Garment:
    id: str
    photo: str
    region: str
    bbox: list[float]            # 원본 사진 좌표 [x0,y0,x1,y1], 0~1 정규화
    area_frac: float             # 사람 crop 대비 면적
    colors: list[dict]
    flags: list[str] = field(default_factory=list)
    crop_file: str = ""
    embedding: np.ndarray | None = None

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "embedding"}
        return d


@dataclass
class PhotoResult:
    photo: str
    status: str                  # ok | no_person | group | no_pose | no_garment | error
    reason: str = ""
    garments: list[Garment] = field(default_factory=list)


def _components(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """4-연결 성분. (면적, 성분 마스크) 큰 순."""
    h, w = mask.shape
    lab = np.zeros((h, w), np.int32)
    comps = []
    n = 0
    for y0, x0 in zip(*np.nonzero(mask)):
        if lab[y0, x0]:
            continue
        n += 1
        q = deque([(y0, x0)])
        lab[y0, x0] = n
        area = 0
        while q:
            y, x = q.popleft()
            area += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = n
                    q.append((yy, xx))
        comps.append((area, n))
    comps.sort(reverse=True)
    return [(a, lab == i) for a, i in comps]


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _vis(pts, *idx, th=0.5):
    return [i for i in idx if pts[i, 2] >= th]


def _mean_y(pts, idx):
    return float(np.mean([pts[i, 1] for i in idx]))


def _refine_split(rgb: np.ndarray, clothes: np.ndarray, y_hip: float, torso: float) -> tuple[float, float]:
    """허리선 근처에서 행별 평균색이 가장 크게 바뀌는 높이를 상·하의 경계로 잡는다.
    반환: (경계 y, 경계 위아래 색 차이). 차이가 작으면 원피스·셋업 후보."""
    h = clothes.shape[0]
    lab = srgb_to_lab(rgb)
    lo, hi = int(max(0, y_hip - 0.45 * torso)), int(min(h - 1, y_hip + 0.3 * torso))
    win = max(3, int(0.08 * torso))
    best, best_y = 0.0, y_hip
    for y in range(lo + win, hi - win):
        a = clothes[y - win:y]
        b = clothes[y:y + win]
        if a.sum() < 30 or b.sum() < 30:
            continue
        ca = lab[y - win:y][a].mean(0)
        cb = lab[y:y + win][b].mean(0)
        d = float(lab_distance(ca, cb))
        if d > best:
            best, best_y = d, float(y)
    return (best_y if best >= 12 else y_hip), best


def _masked_crop(im: Image.Image, mask: np.ndarray, pad: float = 0.06) -> Image.Image:
    x0, y0, x1, y1 = _bbox(mask)
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h))
    x1, y1 = min(im.width, int(x1 + pad * w)), min(im.height, int(y1 + pad * h))
    rgb = np.asarray(im)[y0:y1, x0:x1].copy()
    rgb[~mask[y0:y1, x0:x1]] = NEUTRAL
    side = max(x1 - x0, y1 - y0)
    canvas = Image.new("RGB", (side, side), NEUTRAL)
    canvas.paste(Image.fromarray(rgb), ((side - (x1 - x0)) // 2, (side - (y1 - y0)) // 2))
    return canvas


def analyze_photo(path: Path, vis: V.Vision, cfg: Config, photo_id: str,
                  crops_dir: Path | None = None) -> PhotoResult:
    from .imageio import load_image

    im = load_image(path)
    W, H = im.size

    # 1. 사람 찾기 (저해상도 전체 분할)
    small = im.copy()
    small.thumbnail((256, 256))
    cm = vis.segment(small, out_size=small.size)
    person = cm != V.BG
    comps = [(a, m) for a, m in _components(person) if a >= cfg.min_person_frac * person.size]
    if not comps:
        return PhotoResult(photo_id, "no_person", "사람이 보이지 않음 (옷장 한 컷·플랫레이는 별도 경로)")
    # 사람이 겹치면 한 덩어리가 되므로 얼굴 영역 수도 함께 센다
    # 선글라스·머리카락에 얼굴이 쪼개지지 않도록 팽창 후 센다
    faces = [a for a, _ in _components(_dilate(cm == V.FACE_SKIN, max(2, cm.shape[0] // 40)))
             if a >= cfg.min_face_frac * cm.size]
    n_people = max(
        1 + sum(a >= cfg.group_ratio * comps[0][0] for a, _ in comps[1:]),
        len(faces),
    )
    if n_people > 1 and not cfg.include_group:
        return PhotoResult(photo_id, "group", f"사람 {n_people}명: 누구의 옷인지 알 수 없어 건너뜀")

    # 2. 가장 큰 사람을 정사각 crop
    sx, sy = W / small.width, H / small.height
    x0, y0, x1, y1 = _bbox(comps[0][1])
    cx, cy = (x0 + x1) / 2 * sx, (y0 + y1) / 2 * sy
    side = max((x1 - x0) * sx, (y1 - y0) * sy) * 1.15
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    crop = im.crop(tuple(int(round(v)) for v in box)).resize((cfg.crop_side, cfg.crop_side), Image.LANCZOS)
    scale = side / cfg.crop_side

    presence, pts = vis.pose(crop)
    ccm = vis.segment(crop)
    rgb = np.asarray(crop)
    clothes = ccm == V.CLOTHES
    others = ccm == V.OTHERS
    n = cfg.crop_side
    rows = np.arange(n)[:, None] * np.ones((1, n))

    regions: dict[str, tuple[np.ndarray, list[str]]] = {}
    if presence < cfg.min_presence:
        # 상반신 근접 셀카는 자세 모델이 놓치기 쉽다. 얼굴 아래 옷을 상의로 본다.
        face = ccm == V.FACE_SKIN
        if face.sum() < 0.005 * face.size or clothes.sum() < cfg.min_region_frac * clothes.size:
            return PhotoResult(photo_id, "no_pose", f"자세 인식 실패 (확률 {presence:.2f})")
        fy0, fy1 = np.nonzero(face.any(axis=1))[0][[0, -1]]
        regions["upper"] = (clothes & (rows >= fy0 + 0.6 * (fy1 - fy0)), ["partial", "no_pose"])
        pts = np.zeros((33, 3))

    sh = _vis(pts, V.L_SHOULDER, V.R_SHOULDER, th=cfg.min_vis)
    hp = _vis(pts, V.L_HIP, V.R_HIP, th=cfg.min_vis)
    kn = _vis(pts, V.L_KNEE, V.R_KNEE, th=cfg.min_vis)
    an = _vis(pts, V.L_ANKLE, V.R_ANKLE, V.L_HEEL, V.R_HEEL, V.L_FOOT, V.R_FOOT, th=cfg.min_vis)

    sw = float(abs(pts[V.L_SHOULDER, 0] - pts[V.R_SHOULDER, 0])) if len(sh) == 2 else 0.25 * n
    y_sh = _mean_y(pts, sh) if sh else None
    y_hip = _mean_y(pts, hp) if hp else (y_sh + 1.5 * sw if y_sh is not None else None)
    y_knee = _mean_y(pts, kn) if kn else None
    y_ank = _mean_y(pts, an) if an else None

    split, split_diff = y_hip, 0.0
    if y_sh is not None and y_hip is not None:
        torso = max(8.0, y_hip - y_sh)
        split, split_diff = _refine_split(rgb, clothes, y_hip, torso)
        flags = [] if hp else ["partial"]
        m = clothes & (rows >= y_sh - 0.25 * sw) & (rows < split)
        regions["upper"] = (m, flags)

    if y_hip is not None and y_knee is not None and hp:
        leg = (y_ank - y_knee) if y_ank is not None else (y_knee - y_hip)
        foot_top = (y_ank - 0.25 * leg) if y_ank is not None else n
        m = clothes & (rows >= split) & (rows < foot_top)
        flags = [] if y_ank is not None else ["partial"]
        if split_diff < 12 and "upper" in regions:
            flags.append("maybe_dress")
            regions["upper"][1].append("maybe_dress")
        regions["lower"] = (m, flags)
        if y_ank is not None:
            feet_x = [pts[i, 0] for i in an]
            xl, xr = min(feet_x) - 0.6 * sw, max(feet_x) + 0.6 * sw
            cols = np.ones((n, 1)) * np.arange(n)[None, :]
            m = (clothes | others) & (rows >= foot_top) & (cols >= xl) & (cols <= xr)
            regions["shoes"] = (m, [])

    res = PhotoResult(photo_id, "ok")
    for name in REGIONS:
        if name not in regions:
            continue
        m, flags = regions[name]
        # 가장 큰 성분만 남겨 잡음 제거 (가방끈·배경 오분할)
        if m.sum() < cfg.min_region_frac * m.size:
            continue
        comps_r = _components(np.asarray(Image.fromarray(m).resize((128, 128), Image.NEAREST)))
        if comps_r:
            keep = np.asarray(Image.fromarray(comps_r[0][1]).resize((n, n), Image.NEAREST))
            m = m & keep
        if m.sum() < cfg.min_region_frac * m.size:
            continue
        gcrop = _masked_crop(crop, m)
        bx0, by0, bx1, by1 = _bbox(m)
        bbox = [
            max(0.0, (box[0] + bx0 * scale) / W), max(0.0, (box[1] + by0 * scale) / H),
            min(1.0, (box[0] + bx1 * scale) / W), min(1.0, (box[1] + by1 * scale) / H),
        ]
        g = Garment(
            id=f"{photo_id}#{name}", photo=photo_id, region=name,
            bbox=[round(v, 4) for v in bbox], area_frac=round(float(m.mean()), 4),
            colors=dominant_colors(rgb[m]), flags=sorted(set(flags)),
        )
        g.embedding = vis.embed(gcrop)
        if crops_dir is not None:
            fname = f"{photo_id.replace('/', '_')}__{name}.jpg"
            gcrop.save(crops_dir / fname, quality=88)
            g.crop_file = fname
        res.garments.append(g)
    if not res.garments:
        res.status, res.reason = "no_garment", "옷 영역을 찾지 못함"
    return res
