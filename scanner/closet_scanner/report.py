"""확인용 HTML 보고서. 외부 의존 없는 단일 파일.

"같은 옷인가요?" 질문에 답하면 브라우저에 저장되고, 버튼으로 decisions.json을 내려받아
출력 폴더에 넣은 뒤 `closet-scan organize`를 다시 돌리면 반영된다.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from .color import COLOR_KO

STATUS_KO = {"ok": "사용", "no_person": "사람 없음", "group": "여러 명", "no_pose": "자세 인식 실패",
             "no_garment": "옷 영역 없음", "error": "오류"}
CAT_KO = {"top": "상의", "bottom": "하의", "outer": "아우터", "dress": "원피스", "shoes": "신발",
          "bag": "가방", "hat": "모자", "accessory": "액세서리"}
FLAG_KO = {"partial": "일부만 보임", "maybe_dress": "원피스·셋업 가능성"}

CSS = """
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e3e1dc;--acc:#2b4a8b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#eceae4;--mut:#9a988f;--card:#20201e;--line:#34332f;--acc:#8fb0f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
.mut{color:var(--mut)}.stats{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}.stats b{font-size:24px;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}
.card img{width:100%;aspect-ratio:1;object-fit:contain;background:#808080;border-radius:6px;display:block}
.card .t{font-weight:600;margin-top:6px}.card .s{font-size:12px;color:var(--mut)}
.tag{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:99px;padding:0 6px;margin:2px 2px 0 0}
.thumbs{display:flex;gap:3px;margin-top:6px;flex-wrap:wrap}.thumbs img{width:34px;height:34px;aspect-ratio:auto}
.q{display:flex;gap:12px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin-bottom:10px;flex-wrap:wrap}
.q img{width:150px;height:150px;object-fit:contain;background:#808080;border-radius:6px}
.q button{font:inherit;padding:6px 12px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
.q button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
.bar{position:sticky;bottom:0;background:var(--bg);padding:10px 0;border-top:1px solid var(--line)}
"""

JS = """
const KEY='closet-decisions:'+document.title;
let dec={};try{dec=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function paint(){document.querySelectorAll('.q').forEach(q=>{const v=dec[q.dataset.k];
 q.querySelectorAll('button[data-v]').forEach(b=>b.classList.toggle('on',b.dataset.v===v))});
 document.getElementById('cnt').textContent=Object.keys(dec).length}
document.addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(!b)return;
 const k=b.closest('.q').dataset.k;dec[k]=b.dataset.v;try{localStorage.setItem(KEY,JSON.stringify(dec))}catch(e){}paint()});
function download(){const out=Object.entries(dec).filter(([k,v])=>v!=='skip').map(([k,v])=>{const [a,b]=k.split('|');return{a,b,verdict:v,source:'user'}});
 const blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});const u=URL.createObjectURL(blob);
 const a=document.createElement('a');a.href=u;a.download='decisions.json';a.click();URL.revokeObjectURL(u)}
paint();
"""


def _e(s) -> str:
    return html.escape(str(s))


def write_report(out: Path, title: str = "옷장 스캔 결과") -> Path:
    closet = json.loads((out / "closet.json").read_text())
    photos = json.loads((out / "photos.json").read_text())
    garments = {g["id"]: g for g in json.loads((out / "garments.json").read_text())}
    s = closet["summary"]
    parts = [f"<!doctype html><html lang=ko><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>{_e(title)}</title><style>{CSS}</style></head><body><main>",
             f"<h1>{_e(title)}</h1><p class=mut>사진에서 찾은 옷을 같은 옷끼리 묶은 결과입니다. 얼굴·피부·배경은 회색으로 지워져 있습니다.</p>",
             "<div class=stats>",
             f"<div><b>{s['photos']}</b>사진</div><div><b>{s['photos_used']}</b>옷을 찾은 사진</div>",
             f"<div><b>{s['garment_regions']}</b>옷 영역</div><div><b>{s['items']}</b>옷(묶음)</div>",
             f"<div><b>{s['questions']}</b>확인 질문</div></div>"]

    if closet["questions"]:
        parts.append("<h2>같은 옷인가요?</h2><p class=mut>답하면 이 브라우저에 저장됩니다. 끝나면 아래 버튼으로 decisions.json을 받아 "
                     "스캔 출력 폴더에 넣고 <code>closet-scan organize</code>를 다시 실행하세요.</p>")
        for q in closet["questions"]:
            a, b = garments[q["a"]], garments[q["b"]]
            parts.append(
                f"<div class=q data-k='{_e(q['a'])}|{_e(q['b'])}'><img src='crops/{_e(a['crop_file'])}' alt=''>"
                f"<img src='crops/{_e(b['crop_file'])}' alt=''><div><div class=s>{_e(a['photo'])} ↔ {_e(b['photo'])} · 유사도 {q['score']}</div>"
                "<button data-v=same>같은 옷</button> <button data-v=different>다른 옷</button> <button data-v=skip>모르겠음</button></div></div>")
        parts.append("<div class=bar><button onclick='download()'>decisions.json 내려받기</button> "
                     "<span class=mut>답한 질문 <span id=cnt>0</span>개</span></div>")

    parts.append("<h2>옷장</h2><div class=grid>")
    for it in closet["items"]:
        name = it["name"] or f"{COLOR_KO.get(it['color'], it['color'])} {CAT_KO.get(it['category'], it['category'])}"
        flags = "".join(f"<span class=tag>{_e(FLAG_KO.get(f, f))}</span>" for f in it["flags"])
        thumbs = "".join(f"<img src='crops/{_e(garments[g]['crop_file'])}' alt='' title='{_e(garments[g]['photo'])}'>"
                         for g in it["appearances"][:12])
        parts.append(f"<div class=card><img src='crops/{_e(it['rep_crop'])}' alt=''><div class=t>{_e(name)}</div>"
                     f"<div class=s>{_e(it['item_id'])} · {it['count']}회 등장</div>{flags}<div class=thumbs>{thumbs}</div></div>")
    parts.append("</div>")

    parts.append("<h2>사진별 처리</h2><table><tr><th>사진</th><th>상태</th><th>영역</th><th>사유</th></tr>")
    for pid, p in sorted(photos.items()):
        parts.append(f"<tr><td>{_e(pid)}</td><td>{_e(STATUS_KO.get(p['status'], p['status']))}</td>"
                     f"<td>{p['garments']}</td><td>{_e(p['reason'])}</td></tr>")
    parts.append(f"</table></main><script>{JS}</script></body></html>")
    path = out / "report.html"
    path.write_text("".join(parts), encoding="utf-8")
    return path
