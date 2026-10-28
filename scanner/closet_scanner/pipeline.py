"""스캔 → 묶기 → (선택) 태깅·검증 → closet.json 과 report.html.

출력 폴더 구조
  photos.json      사진별 처리 상태(건너뛴 이유 포함)
  garments.json    옷 영역 목록 (사진·부위·대표색·crop 파일)
  embeddings.npy   garments.json 순서의 임베딩
  crops/           옷 영역 crop (얼굴·배경은 회색)
  decisions.json   "같은 옷인가요?" 답 (사용자·Claude)
  closet.json      최종 옷장: 옷(item)별 등장 사진과 태그
  report.html      확인용 보고서
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .cluster import ClusterConfig, cluster, group_of, similarity_matrix
from .detect import Config, analyze_photo
from .imageio import list_photos

SCHEMA_VERSION = 1


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def _fingerprint(p: Path, detector: str = "parts") -> str:
    st = p.stat()
    base = f"{st.st_size}-{int(st.st_mtime)}"
    return base if detector == "parts" else f"{base}-{detector}"


def scan(photos_dir: Path, out: Path, cfg: Config | None = None, log=print,
         detector: str = "parts", detect_model: str | None = None) -> None:
    """새 사진만 분석해 garments.json·embeddings.npy에 덧붙인다(증분 스캔).

    detector: parts(부위 분할, 기본) 또는 vlm(옷 한 벌 단위 검출, vlm.py).
    """
    from .vision import Vision

    cfg = cfg or Config()
    det = None
    if detector == "vlm":
        from .vlm import VLMDetector

        det = VLMDetector(out / "detect_cache.json", **({"model": detect_model} if detect_model else {}))
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    photos = _read(out / "photos.json", {})
    garments = _read(out / "garments.json", [])
    emb = list(np.load(out / "embeddings.npy")) if (out / "embeddings.npy").exists() else []

    files = list_photos(photos_dir)
    todo = []
    for p in files:
        pid = p.relative_to(photos_dir).as_posix()
        if photos.get(pid, {}).get("fingerprint") != _fingerprint(p, detector):
            todo.append((pid, p))
    log(f"사진 {len(files)}장 중 새로 분석할 사진 {len(todo)}장")
    if not todo:
        return

    redo = {pid for pid, _ in todo}
    keep = [i for i, g in enumerate(garments) if g["photo"] not in redo]
    garments = [garments[i] for i in keep]
    emb = [emb[i] for i in keep]

    vis = Vision()
    t0 = time.time()
    for k, (pid, p) in enumerate(todo, 1):
        try:
            if det is not None:
                from .vlm import analyze_photo_vlm

                r = analyze_photo_vlm(p, vis, det, pid, out / "crops")
            else:
                r = analyze_photo(p, vis, cfg, pid, out / "crops")
        except Exception as e:  # 손상된 파일 하나로 전체가 멈추지 않게
            photos[pid] = {"status": "error", "reason": f"{type(e).__name__}: {e}",
                           "fingerprint": _fingerprint(p, detector), "garments": 0}
            log(f"[{k}/{len(todo)}] {pid}: 오류 {e}")
            continue
        photos[pid] = {"status": r.status, "reason": r.reason, "fingerprint": _fingerprint(p, detector),
                       "garments": len(r.garments)}
        for g in r.garments:
            garments.append(g.to_json())
            emb.append(g.embedding)
        regions = ", ".join(g.tags.get("category", g.region) for g in r.garments) or r.reason
        log(f"[{k}/{len(todo)}] {pid}: {regions}")
    log(f"분석 {len(todo)}장, {time.time() - t0:.1f}초")
    if det is not None:
        log(f"검출 모델 호출 {det.calls}회, 입력 {det.input_tokens} / 출력 {det.output_tokens} 토큰")

    _write(out / "photos.json", photos)
    _write(out / "garments.json", garments)
    arr = np.array(emb, dtype=np.float32) if emb else np.zeros((0, 1280), np.float32)
    np.save(out / "embeddings.npy", arr)


def load(out: Path) -> tuple[list[dict], np.ndarray]:
    garments = _read(out / "garments.json", [])
    emb = np.load(out / "embeddings.npy") if garments else np.zeros((0, 1280), np.float32)
    return garments, emb


def build_items(garments: list[dict], labels: list[int]) -> list[dict]:
    items: dict[int, list[dict]] = {}
    for g, lab in zip(garments, labels):
        items.setdefault(lab, []).append(g)
    out = []
    for lab, members in sorted(items.items(), key=lambda kv: -len(kv[1])):
        rep = max(members, key=lambda g: (("partial" not in g["flags"]), g["area_frac"]))
        tags = rep.get("tags") or {}
        out.append({
            "item_id": f"item{len(out) + 1:03d}",
            "group": group_of(rep),
            "category": tags.get("category") or {"upper": "top", "lower": "bottom", "shoes": "shoes"}[rep["region"]],
            "name": tags.get("name_ko") or "",
            "color": (rep["colors"][0]["name"] if rep["colors"] else ""),
            "count": len(members),
            "photos": sorted({g["photo"] for g in members}),
            "appearances": [g["id"] for g in members],
            "rep_crop": rep.get("crop_file", ""),
            "flags": sorted({f for g in members for f in g["flags"]}),
            "tags": tags,
        })
    return out


def organize(out: Path, ccfg: ClusterConfig | None = None, tagger=None, verify=False, log=print) -> dict:
    """묶기 → (태깅) → (애매한 쌍 Claude 검증 후 다시 묶기) → closet.json."""
    ccfg = ccfg or ClusterConfig()
    garments, emb = load(out)
    decisions = _read(out / "decisions.json", [])
    # 사용자 답이 모델 답보다 우선
    by_pair: dict[tuple[str, str], dict] = {}
    for d in decisions:
        key = tuple(sorted((d["a"], d["b"])))
        if key not in by_pair or d.get("source") == "user" or by_pair[key].get("source") != "user":
            by_pair[key] = d
    active = list(by_pair.values())

    S = similarity_matrix(garments, emb, ccfg)
    labels, questions = cluster(garments, S, ccfg, active)

    if tagger is not None:
        items = build_items(garments, labels)
        log(f"Claude 태깅: 옷 {len(items)}벌")
        tags_by_rep = {}
        for it in items:
            rep = next(g for g in garments if g.get("crop_file") == it["rep_crop"])
            if (rep.get("tags") or {}).get("source") == "vlm":  # 검출 모델이 이미 분류함
                continue
            t = tagger.tag(out / "crops" / rep["crop_file"], rep["region"])
            if t:
                tags_by_rep[it["item_id"]] = t
                for gid in it["appearances"]:
                    next(g for g in garments if g["id"] == gid)["tags"] = t
        _write(out / "garments.json", garments)
        if verify and questions:
            log(f"Claude 쌍 검증: {len(questions)}쌍")
            gmap = {g["id"]: g for g in garments}
            for q in questions:
                v = tagger.same_item(out / "crops" / gmap[q["a"]]["crop_file"], out / "crops" / gmap[q["b"]]["crop_file"])
                if v and v["verdict"] in ("same", "different"):
                    key = tuple(sorted((q["a"], q["b"])))
                    if by_pair.get(key, {}).get("source") != "user":
                        by_pair[key] = {"a": q["a"], "b": q["b"], "verdict": v["verdict"],
                                        "source": "claude", "reason": v["reason"]}
            active = list(by_pair.values())
            _write(out / "decisions.json", active)
            S = similarity_matrix(garments, emb, ccfg)
            labels, questions = cluster(garments, S, ccfg, active)
        log(f"Claude 호출 {tagger.calls}회, 입력 {tagger.input_tokens} / 출력 {tagger.output_tokens} 토큰")

    items = [it for it in build_items(garments, labels) if it["tags"].get("is_garment", True)]
    photos = _read(out / "photos.json", {})
    closet = {
        "version": SCHEMA_VERSION,
        "config": asdict(ccfg),
        "summary": {
            "photos": len(photos),
            "photos_used": sum(1 for p in photos.values() if p["status"] == "ok"),
            "garment_regions": len(garments),
            "items": len(items),
            "questions": len(questions),
        },
        "items": items,
        "questions": questions,
        "garment_item": {gid: it["item_id"] for it in items for gid in it["appearances"]},
    }
    _write(out / "closet.json", closet)
    return closet
