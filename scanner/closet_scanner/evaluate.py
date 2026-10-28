"""M0 평가: 스캐너 결과를 사람이 단 정답 라벨과 비교한다.

라벨 CSV (m0/label.html이 만든다): photo,item,category,color,note
  photo     사진 파일 이름(스캔 폴더 기준 상대 경로도 가능)
  item      옷 번호. 같은 옷은 모든 사진에서 같은 번호
  category  top|bottom|outer|dress|shoes|bag|hat|accessory
  color     기본 색명 키(선택)

지표 정의와 합격 기준은 docs/SCANNER.md 와 m0/README.md.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from .cluster import ClusterConfig, cluster, group_of, similarity_matrix

MAIN = {"top", "bottom", "outer", "dress", "shoes"}
GT_GROUP = {"top": "upper", "outer": "upper", "dress": "upper", "bottom": "lower", "shoes": "shoes"}
# 같은 부위에 정답이 둘 이상일 때 스캐너 영역이 먼저 대응할 순서 (겉에 보이는 옷 우선)
PRIORITY = {"outer": 0, "dress": 1, "top": 2, "bottom": 0, "shoes": 0}
NEAR = [{"gray", "lightgray"}, {"white", "lightgray"}, {"white", "beige"}, {"navy", "denim", "blue"},
        {"skyblue", "blue", "denim"}, {"brown", "beige"}, {"khaki", "green"}, {"mint", "green"},
        {"pink", "red"}, {"black", "navy"}]

PASS = {  # 합격 기준 (사전 등록, 2026-10-28)
    "detection_recall": 0.85,
    "category_accuracy": 0.90,
    "color_accuracy": 0.80,
    "bcubed_f1": 0.80,
    "false_merge_rate": 0.05,   # 이하
    "corrections_per_10": 3.0,  # 이하
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def read_labels(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return [r for r in rows if r.get("photo")]  # item "-" = 옷이 없는 사진 (라벨 완료 표시)


def _photo_key(p: str) -> str:
    return p.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _pred_category(g: dict) -> str:
    return (g.get("tags") or {}).get("category") or {"upper": "top", "lower": "bottom", "shoes": "shoes"}[g["region"]]


def match(garments: list[dict], garment_item: dict, labels: list[dict]):
    """사진별로 스캐너 영역과 정답 옷을 1:1 대응. 반환: (대응 목록, 놓친 정답, 오검출, 범위 밖 정답)."""
    gt_by_photo = defaultdict(list)
    labeled = {_photo_key(r["photo"]) for r in labels}
    out_of_scope = 0
    for r in labels:
        if not r.get("item") or r["item"] == "-":
            continue
        if r["category"] in MAIN:
            gt_by_photo[_photo_key(r["photo"])].append(r)
        else:
            out_of_scope += 1
    pred_by_photo = defaultdict(list)
    for g in garments:
        if g["id"] in garment_item and _photo_key(g["photo"]) in labeled:  # 라벨 안 한 사진은 평가 제외
            pred_by_photo[_photo_key(g["photo"])].append(g)

    pairs, missed, false_pos = [], [], []
    for photo in set(gt_by_photo) | set(pred_by_photo):
        gts = sorted(gt_by_photo[photo], key=lambda r: PRIORITY.get(r["category"], 9))
        used = set()
        for g in pred_by_photo[photo]:
            grp = group_of(g)
            cand = [i for i, r in enumerate(gts) if i not in used and GT_GROUP[r["category"]] == grp]
            if not cand and grp == "lower":  # 원피스의 아랫부분
                cand = [i for i, r in enumerate(gts) if i not in used and r["category"] == "dress"]
            if cand:
                used.add(cand[0])
                pairs.append((g, gts[cand[0]]))
            else:
                false_pos.append(g)
        missed += [r for i, r in enumerate(gts) if i not in used]
    return pairs, missed, false_pos, out_of_scope


def cluster_scores(pairs, garment_item: dict) -> dict:
    pred = [garment_item[g["id"]] for g, _ in pairs]
    gt = [r["item"] for _, r in pairs]
    n = len(pairs)
    if n == 0:
        return {"bcubed_p": float("nan"), "bcubed_r": float("nan"), "bcubed_f1": float("nan"),
                "false_merge_pairs": (0, 0), "split_pairs": (0, 0), "merge_fixes": 0, "split_fixes": 0}
    bp = br = 0.0
    for i in range(n):
        same_pred = [j for j in range(n) if pred[j] == pred[i]]
        same_gt = [j for j in range(n) if gt[j] == gt[i]]
        both = sum(1 for j in same_pred if gt[j] == gt[i])
        bp += both / len(same_pred)
        br += both / len(same_gt)
    bp, br = bp / n, br / n
    fm = fm_tot = sp = sp_tot = 0
    for i, j in itertools.combinations(range(n), 2):
        if pred[i] == pred[j]:
            fm_tot += 1
            fm += gt[i] != gt[j]
        if gt[i] == gt[j]:
            sp_tot += 1
            sp += pred[i] != pred[j]
    clusters_pred = defaultdict(set)
    clusters_gt = defaultdict(set)
    for p, t in zip(pred, gt):
        clusters_pred[p].add(t)
        clusters_gt[t].add(p)
    return {
        "bcubed_p": bp, "bcubed_r": br, "bcubed_f1": 2 * bp * br / (bp + br) if bp + br else 0.0,
        "false_merge_pairs": (fm, fm_tot), "split_pairs": (sp, sp_tot),
        "merge_fixes": sum(len(v) - 1 for v in clusters_pred.values()),
        "split_fixes": sum(len(v) - 1 for v in clusters_gt.values()),
    }


def _color_ok(pred: str, gt: str, lenient: bool) -> bool:
    if pred == gt:
        return True
    return lenient and any(pred in s and gt in s for s in NEAR)


def evaluate(out: Path, labels_path: Path, garment_item: dict | None = None) -> dict:
    garments = json.loads((out / "garments.json").read_text())
    if garment_item is None:
        garment_item = json.loads((out / "closet.json").read_text())["garment_item"]
    labels = read_labels(labels_path)
    pairs, missed, false_pos, oos = match(garments, garment_item, labels)
    n_gt = len(pairs) + len(missed)
    n_pred = len(pairs) + len(false_pos)
    cat_ok = sum(_pred_category(g) == r["category"] for g, r in pairs)
    col_pairs = [(g, r) for g, r in pairs if r.get("color") and g["colors"]]
    col_strict = sum(_color_ok(g["colors"][0]["name"], r["color"], False) for g, r in col_pairs)
    col_len = sum(_color_ok(g["colors"][0]["name"], r["color"], True) for g, r in col_pairs)
    cs = cluster_scores(pairs, garment_item)
    gt_items = {r["item"] for r in labels if r.get("category") in MAIN}
    corrections = len(missed) + len(false_pos) + cs["merge_fixes"] + cs["split_fixes"] + (len(pairs) - cat_ok)
    per10 = 10 * corrections / max(1, len(gt_items))
    fm, fm_tot = cs["false_merge_pairs"]
    sp, sp_tot = cs["split_pairs"]
    return {
        "photos_labeled": len({_photo_key(r["photo"]) for r in labels}),
        "gt_appearances": n_gt, "gt_items": len(gt_items), "out_of_scope_labels": oos,
        "pred_regions": n_pred, "matched": len(pairs),
        "detection_recall": wilson(len(pairs), n_gt),
        "detection_precision": wilson(len(pairs), n_pred),
        "category_accuracy": wilson(cat_ok, len(pairs)),
        "color_accuracy": wilson(col_strict, len(col_pairs)),
        "color_accuracy_lenient": wilson(col_len, len(col_pairs)),
        "bcubed_p": cs["bcubed_p"], "bcubed_r": cs["bcubed_r"], "bcubed_f1": cs["bcubed_f1"],
        "false_merge_rate": wilson(fm, fm_tot), "over_split_rate": wilson(sp, sp_tot),
        "corrections": {"missed": len(missed), "false_positive": len(false_pos),
                        "merge": cs["merge_fixes"], "split": cs["split_fixes"],
                        "category": len(pairs) - cat_ok, "total": corrections},
        "corrections_per_10": per10,
        "missed_examples": [f'{r["photo"]}: {r["item"]} ({r["category"]})' for r in missed[:15]],
        "false_positive_examples": [g["id"] for g in false_pos[:15]],
    }


def sweep(out: Path, labels_path: Path, log=print) -> list[dict]:
    """묶기 기준값 격자 탐색. 같은 데이터로 고르므로 낙관 편향이 있다(보고서에 명시)."""
    import numpy as np

    garments = json.loads((out / "garments.json").read_text())
    emb = np.load(out / "embeddings.npy")
    decisions = [d for d in json.loads((out / "decisions.json").read_text())] if (out / "decisions.json").exists() else []
    results = []
    for w in (0.4, 0.6, 0.8):
        for gate in (20.0, 30.0, 45.0):
            base = ClusterConfig(w_emb=w, color_gate=gate, merge=0.0, ask=0.0)
            S = similarity_matrix(garments, emb, base)
            for merge in [round(x, 3) for x in np.arange(0.50, 0.96, 0.025)]:
                cfg = ClusterConfig(w_emb=w, color_gate=gate, merge=merge, ask=merge)
                labels, _ = cluster(garments, S, cfg, decisions)
                gi = {g["id"]: f"c{lab}" for g, lab in zip(garments, labels)}
                m = evaluate(out, labels_path, gi)
                results.append({"w_emb": w, "color_gate": gate, "merge": merge,
                                "bcubed_f1": m["bcubed_f1"], "false_merge": m["false_merge_rate"][0],
                                "over_split": m["over_split_rate"][0], "per10": m["corrections_per_10"]})
    ok = [r for r in results if not (r["false_merge"] > PASS["false_merge_rate"])]
    ok.sort(key=lambda r: (-(r["bcubed_f1"] if r["bcubed_f1"] == r["bcubed_f1"] else -1), r["per10"]))
    return ok[:10]


def _fmt(v) -> str:
    if isinstance(v, tuple):
        p, lo, hi = v
        return "—" if p != p else f"{100 * p:.1f}% [{100 * lo:.1f}–{100 * hi:.1f}]"
    if isinstance(v, float):
        return "—" if v != v else f"{v:.3f}"
    return str(v)


def to_markdown(m: dict, title: str = "M0 평가 결과") -> str:
    def verdict(key, val, lower_better=False):
        th = PASS[key]
        x = val[0] if isinstance(val, tuple) else val
        if x != x:
            return "판정 불가"
        return "합격" if (x <= th if lower_better else x >= th) else "불합격"

    rows = [
        ("검출 재현율", "detection_recall", False, "≥ 85%"),
        ("대분류 정확도", "category_accuracy", False, "≥ 90%"),
        ("색 정확도", "color_accuracy", False, "≥ 80%"),
        ("같은 옷 판별 B-cubed F1", "bcubed_f1", False, "≥ 0.80"),
        ("과병합률(다른 옷을 한 벌로)", "false_merge_rate", True, "≤ 5%"),
        ("수정 부담(옷 10벌당)", "corrections_per_10", True, "≤ 3"),
    ]
    lines = [f"# {title}", "",
             f"라벨 사진 {m['photos_labeled']}장, 정답 등장 {m['gt_appearances']}회, 실제 옷 {m['gt_items']}벌, "
             f"스캐너 영역 {m['pred_regions']}개, 대응 {m['matched']}개. 괄호는 95% Wilson 신뢰구간.", "",
             "| 지표 | 값 | 기준 | 판정 |", "| --- | --- | --- | --- |"]
    for name, key, lower, th in rows:
        lines.append(f"| {name} | {_fmt(m[key])} | {th} | {verdict(key, m[key], lower)} |")
    c = m["corrections"]
    lines += ["", "| 보조 지표 | 값 |", "| --- | --- |",
              f"| 검출 정밀도 | {_fmt(m['detection_precision'])} |",
              f"| 색 정확도(인접 색 허용) | {_fmt(m['color_accuracy_lenient'])} |",
              f"| B-cubed 정밀도 / 재현율 | {_fmt(m['bcubed_p'])} / {_fmt(m['bcubed_r'])} |",
              f"| 과분할률(한 벌을 여러 벌로) | {_fmt(m['over_split_rate'])} |",
              f"| 수정 내역 | 놓침 {c['missed']}, 오검출 {c['false_positive']}, 합치기 {c['merge']}, "
              f"나누기 {c['split']}, 분류 {c['category']} |",
              f"| 범위 밖 라벨(가방·모자·액세서리) | {m['out_of_scope_labels']} |"]
    if m["missed_examples"]:
        lines += ["", "놓친 예: " + "; ".join(m["missed_examples"])]
    if m["false_positive_examples"]:
        lines += ["", "오검출 예: " + "; ".join(m["false_positive_examples"])]
    return "\n".join(lines) + "\n"


def oracle_questions(out: Path, labels_path: Path, cfg: ClusterConfig | None = None, rounds: int = 3) -> list[dict]:
    """사용자가 "같은 옷인가요?" 질문에 정답대로 답한다고 가정하고, 질문 수에 따른 품질 변화를 본다.
    질문 큐 설계(과분할 우선 + 확인)가 실제로 수정 부담을 줄이는지 검증하는 용도."""
    import numpy as np

    cfg = cfg or ClusterConfig()
    garments = json.loads((out / "garments.json").read_text())
    emb = np.load(out / "embeddings.npy")
    labels = read_labels(labels_path)
    S = similarity_matrix(garments, emb, cfg)
    decisions: list[dict] = []
    curve = []
    asked = 0
    for r in range(rounds + 1):
        lab, questions = cluster(garments, S, cfg, decisions)
        gi = {g["id"]: f"c{x}" for g, x in zip(garments, lab)}
        m = evaluate(out, labels_path, gi)
        curve.append({"questions": asked, "bcubed_f1": m["bcubed_f1"], "false_merge": m["false_merge_rate"][0],
                      "per10": m["corrections_per_10"]})
        if r == rounds or not questions:
            break
        pairs, *_ = match(garments, gi, labels)
        truth = {g["id"]: t["item"] for g, t in pairs}
        for q in questions:
            if q["a"] in truth and q["b"] in truth:
                decisions.append({"a": q["a"], "b": q["b"], "source": "user",
                                  "verdict": "same" if truth[q["a"]] == truth[q["b"]] else "different"})
                asked += 1
    return curve
