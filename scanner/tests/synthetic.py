"""합성 갤러리: 같은 옷이 여러 사진에 나오는 상황을 증강으로 흉내 낸다.

실제 M0(윤경 사진 50장)의 대체가 아니다. 거울 반전·구도·조명·화이트밸런스가 바뀌어도
같은 옷으로 묶이는지(불변성)와, 다른 사진의 옷을 합치지 않는지(과병합)를 본다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closet_scanner.imageio import list_photos, load_image  # noqa: E402


def _wb(im: Image.Image, r: float, b: float) -> Image.Image:
    a = np.asarray(im, np.float32)
    a[..., 0] *= r
    a[..., 2] *= b
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


VARIANTS = {
    "a_orig": lambda im: im,
    "b_mirror": lambda im: ImageOps.mirror(im),
    "c_crop": lambda im: im.crop((int(im.width * 0.04), int(im.height * 0.03), int(im.width * 0.97), im.height)),
    "d_warm_dark": lambda im: ImageEnhance.Brightness(_wb(im, 1.08, 0.88)).enhance(0.85),
    "e_cool_bright": lambda im: ImageEnhance.Brightness(_wb(im, 0.92, 1.08)).enhance(1.12).rotate(2.5, fillcolor=(235, 235, 235)),
}


def build(sources: list[Path], dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for src in sources:
        im = load_image(src, 1400)
        for key, fn in VARIANTS.items():
            name = f"{src.stem}__{key}.jpg"
            fn(im).convert("RGB").save(dst / name, quality=90)
            names.append(name)
    return names


def write_labels(out_dir: Path, labels_csv: Path) -> None:
    """원본(a_orig)에서 찾은 영역을 정답 옷으로 삼아, 모든 변형에 같은 번호를 붙인다."""
    import json

    garments = json.loads((out_dir / "garments.json").read_text())
    cat = {"upper": "top", "lower": "bottom", "shoes": "shoes"}
    base = {(g["photo"].split("__")[0], g["region"]) for g in garments if "__a_orig" in g["photo"]}
    photos = sorted(json.loads((out_dir / "photos.json").read_text()))
    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo", "item", "category", "color", "note"])
        for p in photos:
            stem = p.split("__")[0]
            for s, region in sorted(base):
                if s == stem:
                    w.writerow([p, f"{stem}:{region}", cat[region], "", "synthetic"])


if __name__ == "__main__":
    src_dir, dst = Path(sys.argv[1]), Path(sys.argv[2])
    picks = [p for p in list_photos(src_dir) if not sys.argv[3:] or p.stem in sys.argv[3:]]
    print(len(build(picks, dst)), "장 생성")
