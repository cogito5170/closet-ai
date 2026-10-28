"""옷 한 벌 단위 경로: bbox 대응 규칙, 내보내기, VLM 검출(가짜 모델)."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closet_scanner.bbox_eval import best_match, evaluate_bbox, iou
from closet_scanner.vlm import clean_bbox


def test_iou():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 0.5, 1], [0.5, 0, 1, 1]) == 0.0
    assert abs(iou([0, 0, 1, 1], [0, 0, 0.5, 1]) - 0.5) < 1e-9


def test_match_maximizes_count_before_iou():
    # 탐욕 대응이면 p0-g0(0.9)을 먼저 잡아 1쌍에 그친다. 개수 우선이면 2쌍.
    gt = [[0, 0, 1.0, 1.0], [0, 0, 0.6, 1.0]]
    pred = [[0, 0, 0.9, 1.0], [0.3, 0, 1.0, 1.0]]
    pairs = best_match(pred, gt, 0.5)
    assert len(pairs) == 2


def test_evaluate_bbox(tmp_path):
    gt = {"photos": [{"photo": "a.jpg", "garments": [
        {"category": "dress", "bbox": [0.2, 0.2, 0.6, 0.8]},
        {"category": "shoes", "bbox": [0.2, 0.9, 0.3, 1.0]},
        {"category": "shoes", "bbox": [0.5, 0.9, 0.6, 1.0]},
        {"category": "other", "bbox": [0, 0, 1, 1]}]}]}
    pred = {"photos": [{"photo": "a.jpg", "garments": [
        {"category": "top", "bbox": [0.2, 0.2, 0.6, 0.5]},
        {"category": "dress", "bbox": [0.2, 0.2, 0.6, 0.78]},
        {"category": "shoes", "bbox": [0.2, 0.9, 0.6, 1.0]}]}]}
    (tmp_path / "g.json").write_text(json.dumps(gt))
    (tmp_path / "p.json").write_text(json.dumps(pred))
    m = evaluate_bbox(tmp_path / "p.json", tmp_path / "g.json", (0.5,))["by_iou"]["0.5"]
    assert (m["matched"], m["pred"], m["gt"]) == (2, 3, 2)  # 신발 두 짝은 한 켤레, other 제외
    assert m["class_acc_matched"][0] == 1.0
    assert m["recall_by_category"]["dress"] == {"matched": 1, "gt": 1}


def test_clean_bbox():
    assert clean_bbox([0.6, 0.8, 0.2, 0.1]) == [0.2, 0.1, 0.6, 0.8]
    assert clean_bbox([-0.1, 0, 1.2, 0.5]) == [0.0, 0.0, 1.0, 0.5]
    assert clean_bbox([0.5, 0.5, 0.5, 0.9]) is None
    assert clean_bbox(None) is None


class FakeDetector:
    calls = input_tokens = output_tokens = 0

    def detect(self, im):
        return {"status": "ok", "garments": [
            {"category": "outer", "name_ko": "빨강 코트", "color": "빨강", "bbox": [0.1, 0.1, 0.9, 0.7],
             "confidence": 0.9, "note": ""},
            {"category": "top", "name_ko": "파랑 티", "color": "파랑", "bbox": [0.35, 0.15, 0.65, 0.6],
             "confidence": 0.4, "note": "가려짐"},
            {"category": "shoes", "name_ko": "", "color": "검정", "bbox": [0.3, 0.9, 0.3, 0.95],
             "confidence": 0.9, "note": ""}]}


@pytest.fixture(scope="module")
def vis():
    try:
        from closet_scanner.models import ensure_all
        from closet_scanner.vision import Vision
        ensure_all()
        return Vision()
    except Exception as e:  # 오프라인 환경
        pytest.skip(f"모델을 준비할 수 없음: {e}")


def test_vlm_photo_to_garments(vis, tmp_path):
    from closet_scanner.bbox_eval import export_predictions
    from closet_scanner.vlm import analyze_photo_vlm

    im = Image.new("RGB", (400, 600), (230, 230, 230))
    im.paste((200, 30, 30), (40, 60, 360, 420))
    im.paste((30, 60, 200), (140, 90, 260, 360))
    p = tmp_path / "x.jpg"
    im.save(p)
    r = analyze_photo_vlm(p, vis, FakeDetector(), "x.jpg", tmp_path)
    assert r.status == "ok"
    assert [g.tags["category"] for g in r.garments] == ["outer", "top"]  # 넓이 없는 신발 상자는 버림
    assert [g.region for g in r.garments] == ["upper", "upper"]
    assert r.garments[0].colors[0]["name"] == "red"  # 안쪽 상의 영역은 아우터 색에서 뺀다
    assert r.garments[1].colors[0]["name"] == "blue"
    assert "low_confidence" in r.garments[1].flags
    assert r.garments[0].embedding.shape == (1280,)

    (tmp_path / "photos.json").write_text(json.dumps({"x.jpg": {"status": "ok"}}))
    (tmp_path / "garments.json").write_text(json.dumps([g.to_json() for g in r.garments], ensure_ascii=False))
    ex = export_predictions(tmp_path)
    assert [g["category"] for g in ex["photos"][0]["garments"]] == ["outer", "top"]


def test_m0_match_prefers_same_category():
    from closet_scanner.evaluate import match

    def g(i, cat, region):
        return {"id": f"a.jpg#{i}", "photo": "a.jpg", "region": region, "tags": {"category": cat}}

    garments = [g(1, "top", "upper"), g(2, "outer", "upper"), g(3, "other", "other")]
    labels = [{"photo": "a.jpg", "item": "coat", "category": "outer"},
              {"photo": "a.jpg", "item": "tee", "category": "top"}]
    pairs, missed, fp, _ = match(garments, {x["id"]: "i" for x in garments}, labels)
    assert {(p["tags"]["category"], r["category"]) for p, r in pairs} == {("top", "top"), ("outer", "outer")}
    assert not missed and not fp  # other는 오검출로 세지 않는다
