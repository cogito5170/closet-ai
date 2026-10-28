"""모델 없이 도는 단위 테스트: 색, 묶기 규칙, 평가 지표."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closet_scanner.cluster import ClusterConfig, cluster, similarity_matrix  # noqa: E402
from closet_scanner.color import color_name, dominant_colors, palette_distance, srgb_to_lab  # noqa: E402
from closet_scanner.evaluate import evaluate, wilson  # noqa: E402


def test_lab_reference_values():
    lab = srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]]))
    assert lab[0] == pytest.approx([100, 0, 0], abs=0.5)
    assert lab[1] == pytest.approx([0, 0, 0], abs=0.5)
    assert lab[2] == pytest.approx([53.2, 80.1, 67.2], abs=0.5)


@pytest.mark.parametrize("rgb,name", [((20, 20, 24), "black"), ((30, 40, 75), "navy"), ((245, 245, 240), "white"),
                                      ((200, 30, 40), "red"), ((215, 195, 160), "beige")])
def test_color_names(rgb, name):
    assert color_name(srgb_to_lab(np.array(rgb))) == name


def test_dominant_colors_merges_same_name():
    px = np.concatenate([np.full((600, 3), (20, 20, 24)), np.full((500, 3), (35, 35, 38)), np.full((300, 3), (240, 240, 235))])
    cols = dominant_colors(px.astype(np.uint8))
    assert [c["name"] for c in cols] == ["black", "white"]
    assert cols[0]["frac"] == pytest.approx(1100 / 1400, abs=0.02)


def _g(gid, photo, region="upper", rgb=(30, 40, 75)):
    lab = [round(float(v), 2) for v in srgb_to_lab(np.array(rgb))]
    return {"id": gid, "photo": photo, "region": region, "colors": [{"lab": lab, "frac": 1.0, "name": "x"}],
            "flags": [], "area_frac": 0.1, "crop_file": gid + ".jpg"}


def _emb(*vecs):
    e = np.array(vecs, dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_same_photo_never_merges():
    g = [_g("a", "p1"), _g("b", "p1")]
    S = similarity_matrix(g, _emb([1, 0], [1, 0]), ClusterConfig())
    labels, _ = cluster(g, S, ClusterConfig())
    assert labels[0] != labels[1]


def test_color_gate_blocks_merge():
    g = [_g("a", "p1", rgb=(20, 20, 24)), _g("b", "p2", rgb=(245, 245, 240))]
    S = similarity_matrix(g, _emb([1, 0], [1, 0]), ClusterConfig())
    assert S[0, 1] == -1


def test_complete_linkage_prevents_chaining():
    # a~b, b~c 이지만 a와 c는 다르다 → 셋을 한 벌로 합치지 않는다
    g = [_g("a", "p1"), _g("b", "p2"), _g("c", "p3")]
    e = _emb([1, 0], [np.cos(0.5), np.sin(0.5)], [np.cos(1.0), np.sin(1.0)])
    cfg = ClusterConfig(w_emb=1.0, merge=0.85)
    labels, _ = cluster(g, similarity_matrix(g, e, cfg), cfg)
    assert len(set(labels)) == 2


def test_user_decisions_override_model():
    g = [_g("a", "p1"), _g("b", "p2"), _g("c", "p3")]
    e = _emb([1, 0], [1, 0.05], [0.2, 1])
    cfg = ClusterConfig(w_emb=1.0, merge=0.9, ask=0.1)
    S = similarity_matrix(g, e, cfg)
    labels, _ = cluster(g, S, cfg)
    assert labels[0] == labels[1] and labels[2] != labels[0]
    dec = [{"a": "a", "b": "b", "verdict": "different", "source": "user"},
           {"a": "a", "b": "c", "verdict": "same", "source": "user"}]
    labels, questions = cluster(g, S, cfg, dec)
    assert labels[0] != labels[1] and labels[0] == labels[2]


def test_question_queue_ranks_split_fragments_first():
    # x1,x2,x3는 같은 옷이지만 x3만 약간 멀다 → 병합 기준 아래, 질문 1순위
    g = [_g("x1", "p1"), _g("x2", "p2"), _g("x3", "p3"), _g("y", "p4")]
    e = _emb([1, 0, 0], [0.99, 0.1, 0], [0.8, 0.3, 0.5], [0.5, 0, 0.87])
    cfg = ClusterConfig(w_emb=1.0, merge=0.95, ask=0.4)
    labels, q = cluster(g, similarity_matrix(g, e, cfg), cfg)
    assert labels[0] == labels[1] != labels[2]
    assert {q[0]["a"], q[0]["b"]} & {"x3"}


def test_wilson_interval():
    p, lo, hi = wilson(135, 150)
    assert p == pytest.approx(0.9)
    assert lo == pytest.approx(0.842, abs=0.001) and hi == pytest.approx(0.938, abs=0.001)


def test_evaluate_counts(tmp_path):
    garments = [
        {**_g("p1#upper", "p1.jpg"), "region": "upper"},
        {**_g("p1#lower", "p1.jpg"), "region": "lower"},
        {**_g("p2#upper", "p2.jpg"), "region": "upper"},
        {**_g("p3#upper", "p3.jpg"), "region": "upper"},
    ]
    (tmp_path / "garments.json").write_text(json.dumps(garments))
    # 스캐너: p1·p2 상의를 한 벌, p3 상의를 다른 벌로 (정답은 셋 다 T1 → 과분할 1)
    gi = {"p1#upper": "i1", "p2#upper": "i1", "p3#upper": "i2", "p1#lower": "i3"}
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "item", "category", "color", "note"])
        for p in ("p1.jpg", "p2.jpg", "p3.jpg"):
            w.writerow([p, "T1", "top", "", ""])
        w.writerow(["p1.jpg", "B1", "bottom", "", ""])
        w.writerow(["p2.jpg", "S1", "shoes", "", ""])   # 놓친 신발
        w.writerow(["p2.jpg", "H1", "hat", "", ""])     # 범위 밖
    m = evaluate(tmp_path, tmp_path / "labels.csv", gi)
    assert m["matched"] == 4 and m["corrections"]["missed"] == 1 and m["out_of_scope_labels"] == 1
    assert m["false_merge_rate"][0] == 0.0
    assert m["corrections"]["split"] == 1 and m["corrections"]["merge"] == 0
    assert m["bcubed_p"] == pytest.approx(1.0)
