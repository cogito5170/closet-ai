"""모델을 내려받아 실제 파이프라인을 돌리는 통합 테스트.

CLOSET_MODELS_DIR에 모델이 있거나 네트워크가 되면 돈다. 사람 사진 대신 합성 그림으로
'사람 없음' 경로와 출력 파일 구조를 확인한다(실사진 품질은 M0에서 본다).
"""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def models():
    try:
        from closet_scanner.models import ensure_all
        ensure_all()
    except Exception as e:  # 오프라인 환경
        pytest.skip(f"모델을 준비할 수 없음: {e}")


def test_scan_outputs(models, tmp_path):
    from closet_scanner.cli import main

    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (640, 480), (200, 180, 160)).save(photos / "wall.jpg")
    (photos / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "out"
    assert main(["scan", str(photos), "--out", str(out)]) == 0
    status = json.loads((out / "photos.json").read_text())
    assert status["wall.jpg"]["status"] == "no_person"
    assert status["broken.jpg"]["status"] == "error"
    closet = json.loads((out / "closet.json").read_text())
    assert closet["summary"]["items"] == 0
    assert (out / "report.html").exists()
    # 다시 돌리면 바뀐 사진이 없어 분석을 건너뛴다
    assert main(["scan", str(photos), "--out", str(out)]) == 0
