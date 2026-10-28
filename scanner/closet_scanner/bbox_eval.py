"""옷 한 벌 단위 bbox 검출 평가와 예측 파일 변환.

예측과 정답은 같은 형식(closet-crosscheck-v1 predictions.json)을 쓴다.
  {"photos": [{"photo": "photo_001.jpg", "garments": [{"category": "dress", "bbox": [x0,y0,x1,y1]}, ...]}]}

대응 규칙(교차 검증 보고서와 같음):
- 사진마다 클래스 무관 1:1 대응. IoU ≥ 기준인 쌍만 후보.
- 대응 개수를 먼저 최대화하고, 같으면 IoU 합을 최대화한다.
- 대상은 top/bottom/outer/dress/shoes. other(점프수트·케이프)는 양쪽에서 제외한다.
  예측의 unknown은 검출로 세되 분류는 틀린 것으로 본다.
- 분류 정확도는 대응된 쌍에서만 계산한다. 전체 옷의 분류 성공률이 아니다.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from .evaluate import wilson

MAIN = ("top", "bottom", "outer", "dress", "shoes")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best_match(pred: list, gt: list, th: float) -> list[tuple[int, int, float]]:
    """개수 우선, 다음 IoU 합 최대의 1:1 대응. 사진당 옷이 적어 부분집합 DP로 정확히 푼다."""
    M = [[iou(p, g) for g in gt] for p in pred]

    @cache
    def go(i: int, used: int) -> tuple[int, float, tuple]:
        if i == len(pred):
            return 0, 0.0, ()
        best = go(i + 1, used)
        for j in range(len(gt)):
            if not used >> j & 1 and M[i][j] >= th:
                n, s, pairs = go(i + 1, used | 1 << j)
                cand = (n + 1, s + M[i][j], ((i, j, M[i][j]),) + pairs)
                if cand[:2] > best[:2]:
                    best = cand
        return best

    return list(go(0, 0)[2])


def _load(path: Path) -> dict[str, list[dict]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for p in d["photos"]:
        key = p["photo"].replace("\\", "/").rsplit("/", 1)[-1].lower()
        out[key] = [g for g in p.get("garments", []) if g.get("bbox") and g.get("category") != "other"]
    return out


def _merge_shoes(gs: list[dict]) -> list[dict]:
    shoes = [g for g in gs if g["category"] == "shoes"]
    if len(shoes) < 2:
        return gs
    b = [min(g["bbox"][0] for g in shoes), min(g["bbox"][1] for g in shoes),
         max(g["bbox"][2] for g in shoes), max(g["bbox"][3] for g in shoes)]
    return [g for g in gs if g["category"] != "shoes"] + [{"category": "shoes", "bbox": b}]


def evaluate_bbox(pred_path: Path, gt_path: Path, thresholds=(0.5, 0.25), merge_shoes=True) -> dict:
    pred, gt = _load(pred_path), _load(gt_path)
    gt = {k: [g for g in v if g["category"] in MAIN] for k, v in gt.items()}
    if merge_shoes:
        pred = {k: _merge_shoes(v) for k, v in pred.items()}
        gt = {k: _merge_shoes(v) for k, v in gt.items()}
    photos = sorted(gt)  # 정답이 있는 사진만 평가
    res = {"photos": len(photos), "missing_pred_photos": [p for p in photos if p not in pred], "by_iou": {}}
    for th in thresholds:
        tp = n_pred = n_gt = cls_ok = clean = 0
        per_cat = {c: [0, 0] for c in MAIN}  # [대응, 정답 수]
        for ph in photos:
            P, G = pred.get(ph, []), gt[ph]
            pairs = best_match([g["bbox"] for g in P], [g["bbox"] for g in G], th)
            tp += len(pairs)
            n_pred += len(P)
            n_gt += len(G)
            cls_ok += sum(P[i]["category"] == G[j]["category"] for i, j, _ in pairs)
            clean += len(pairs) == len(P) == len(G)
            for g in G:
                per_cat[g["category"]][1] += 1
            for _, j, _ in pairs:
                per_cat[G[j]["category"]][0] += 1
        prec, rec = wilson(tp, n_pred), wilson(tp, n_gt)
        f1 = 2 * prec[0] * rec[0] / (prec[0] + rec[0]) if tp else 0.0
        res["by_iou"][str(th)] = {
            "matched": tp, "pred": n_pred, "gt": n_gt,
            "precision": prec, "recall": rec, "f1": f1,
            "class_acc_matched": wilson(cls_ok, tp),
            "photos_exact": clean,
            "recall_by_category": {c: {"matched": m, "gt": n} for c, (m, n) in per_cat.items()},
        }
    return res


def to_markdown(r: dict) -> str:
    pct = lambda w: f"{100 * w[0]:.1f}% ({100 * w[1]:.0f}–{100 * w[2]:.0f})"
    lines = ["# bbox 검출 평가", "",
             (f"정답이 있는 사진 {r['photos']}장. 괄호는 95% Wilson 신뢰구간. "
              "분류 정확도는 대응된 옷에서만 계산했다."), "",
             "| IoU 기준 | 정밀도 | 재현율 | F1 | 대응 옷의 분류 정확도 | 누락·추가 없는 사진 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for th, m in r["by_iou"].items():
        lines.append(f"| ≥ {th} | {pct(m['precision'])} | {pct(m['recall'])} | {m['f1']:.3f} | "
                     f"{pct(m['class_acc_matched'])} | {m['photos_exact']}/{r['photos']} |")
    lines += ["", "## 종류별 재현율", "", "| 종류 | " + " | ".join(f"IoU ≥ {t}" for t in r["by_iou"]) + " |",
              "| --- |" + " --- |" * len(r["by_iou"])]
    for c in MAIN:
        cells = []
        for m in r["by_iou"].values():
            x = m["recall_by_category"][c]
            cells.append(f"{x['matched']}/{x['gt']}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    if r["missing_pred_photos"]:
        lines += ["", f"예측이 없는 사진 {len(r['missing_pred_photos'])}장: " + ", ".join(r["missing_pred_photos"])]
    return "\n".join(lines) + "\n"


def export_predictions(out: Path) -> dict:
    """스캐너 출력 폴더 → 교차 검증 형식 predictions.json 내용."""
    photos = json.loads((out / "photos.json").read_text())
    garments = json.loads((out / "garments.json").read_text())
    by_photo: dict[str, list] = {pid: [] for pid in photos}
    region_cat = {"upper": "top", "lower": "bottom", "shoes": "shoes"}
    for g in garments:
        tags = g.get("tags") or {}
        by_photo.setdefault(g["photo"], []).append({
            "local_id": g["id"].split("#", 1)[-1],
            "category": tags.get("category") or region_cat.get(g["region"], "unknown"),
            "color": tags.get("color") or (g["colors"][0]["name"] if g["colors"] else ""),
            "bbox": g["bbox"],
            "confidence": tags.get("confidence"),
            "note": tags.get("note", ""),
        })
    status_map = {"ok": "ok", "group": "group", "no_person": "no_person"}
    return {"photos": [{"photo": pid, "status": status_map.get(photos.get(pid, {}).get("status"), "uncertain"),
                        "garments": gs} for pid, gs in sorted(by_photo.items())]}
