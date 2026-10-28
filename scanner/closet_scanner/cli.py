"""closet-scan 명령.

  closet-scan scan  <사진폴더> --out <출력>   분석 + 묶기 + 보고서
  closet-scan organize <출력>                결정(decisions.json) 반영해 다시 묶기
  closet-scan eval  <출력> <labels.csv>      M0 지표 (--sweep: 기준값 탐색)
  closet-scan models                         모델만 미리 내려받기
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cluster import ClusterConfig
from .detect import Config


def _cluster_cfg(a) -> ClusterConfig:
    c = ClusterConfig()
    for k in ("w_emb", "color_gate", "merge", "ask"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(c, k, v)
    return c


def _tagger(a, out: Path):
    if a.tagger == "none":
        return None
    from .tagger import ClaudeTagger

    return ClaudeTagger(out / "tag_cache.json", model=a.model) if a.model else ClaudeTagger(out / "tag_cache.json")


def _organize(a, out: Path) -> None:
    from .pipeline import organize
    from .report import write_report

    closet = organize(out, _cluster_cfg(a), tagger=_tagger(a, out), verify=a.verify)
    rep = write_report(out)
    s = closet["summary"]
    print(f"옷 {s['items']}벌 (영역 {s['garment_regions']}개, 사진 {s['photos_used']}/{s['photos']}장 사용), "
          f"확인 질문 {s['questions']}개")
    print(f"보고서: {rep}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="closet-scan", description="갤러리 사진으로 옷장 만들기")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cluster_args(p):
        p.add_argument("--merge", type=float, help="같은 옷으로 합칠 유사도 기준 (기본 0.60)")
        p.add_argument("--ask", type=float, help="질문 큐에 올릴 유사도 기준 (기본 0.48)")
        p.add_argument("--w-emb", dest="w_emb", type=float, help="임베딩 가중치 (기본 0.6)")
        p.add_argument("--color-gate", dest="color_gate", type=float, help="색 거리 상한 (기본 30)")
        p.add_argument("--tagger", choices=["none", "claude"], default="none", help="Claude 태깅 사용 여부")
        p.add_argument("--verify", action="store_true", help="애매한 쌍을 Claude로 검증 (--tagger claude 필요)")
        p.add_argument("--model", help="Claude 모델 (기본 claude-haiku-5-5)")

    p = sub.add_parser("scan", help="사진 폴더를 분석해 옷장을 만든다")
    p.add_argument("photos", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--include-group", action="store_true", help="여러 명 사진에서도 가장 큰 사람의 옷을 쓴다")
    cluster_args(p)

    p = sub.add_parser("organize", help="decisions.json을 반영해 다시 묶는다")
    p.add_argument("out", type=Path)
    cluster_args(p)

    p = sub.add_parser("eval", help="M0 정답 라벨과 비교")
    p.add_argument("out", type=Path)
    p.add_argument("labels", type=Path)
    p.add_argument("--sweep", action="store_true", help="묶기 기준값 격자 탐색")
    p.add_argument("--md", type=Path, help="결과를 마크다운으로 저장")

    sub.add_parser("models", help="모델 미리 내려받기")

    a = ap.parse_args(argv)
    if a.cmd == "models":
        from .models import ensure_all, models_dir

        ensure_all()
        print(f"모델 위치: {models_dir()}")
    elif a.cmd == "scan":
        if a.verify and a.tagger != "claude":
            ap.error("--verify는 --tagger claude와 함께 써야 합니다")
        from .pipeline import scan

        scan(a.photos, a.out, Config(include_group=a.include_group))
        _organize(a, a.out)
    elif a.cmd == "organize":
        if a.verify and a.tagger != "claude":
            ap.error("--verify는 --tagger claude와 함께 써야 합니다")
        _organize(a, a.out)
    elif a.cmd == "eval":
        from .evaluate import evaluate, oracle_questions, sweep, to_markdown

        m = evaluate(a.out, a.labels)
        md = to_markdown(m)
        curve = oracle_questions(a.out, a.labels)
        md += "\n## 질문에 답하면 (정답대로 답한다고 가정)\n\n| 답한 질문 | B-cubed F1 | 과병합 | 10벌당 수정 |\n| --- | --- | --- | --- |\n"
        for r in curve:
            md += f"| {r['questions']} | {r['bcubed_f1']:.3f} | {100 * r['false_merge']:.1f}% | {r['per10']:.1f} |\n"
        if a.sweep:
            top = sweep(a.out, a.labels)
            md += "\n## 기준값 탐색 (상위 10, 과병합 5% 이하)\n\n같은 데이터로 고른 값이라 낙관적이다. 다른 사람 사진으로 다시 확인해야 한다.\n\n"
            md += "| w_emb | color_gate | merge | B-cubed F1 | 과병합 | 과분할 | 10벌당 수정 |\n| --- | --- | --- | --- | --- | --- | --- |\n"
            for r in top:
                md += (f"| {r['w_emb']} | {r['color_gate']} | {r['merge']} | {r['bcubed_f1']:.3f} | "
                       f"{100 * r['false_merge']:.1f}% | {100 * r['over_split']:.1f}% | {r['per10']:.1f} |\n")
        print(md)
        if a.md:
            a.md.write_text(md, encoding="utf-8")
        (a.out / "eval.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
