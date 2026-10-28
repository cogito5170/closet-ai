"""사진 읽기. HEIC(iPhone 기본 형식)를 포함하고 EXIF 회전을 바로잡는다."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # HEIC 없이도 나머지 형식은 동작
    pass

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def list_photos(folder: Path) -> list[Path]:
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in PHOTO_EXTS and not p.name.startswith(".")
    )


def load_image(path: Path, max_side: int = 1600) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im
