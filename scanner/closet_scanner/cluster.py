"""같은 옷 묶기.

원칙(docs/SCANNER.md):
- 과병합(다른 옷을 한 벌로)이 과분할보다 비싸다. 그래서 완전 연결(complete linkage):
  두 묶음은 모든 교차 쌍이 기준 이상일 때만 합친다.
- 같은 사진의 두 영역은 절대 같은 옷이 아니다(cannot-link).
- 사용자 확인(decisions)은 모델보다 우선한다: same → must-link, different → cannot-link.
- 애매한 묶음 쌍은 "같은 옷인가요?" 질문 큐로 보낸다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import palette_distance

# 태거가 붙인 세부 분류를 비교 가능한 부위로 묶는다
CATEGORY_GROUP = {
    "top": "upper", "outer": "upper", "dress": "upper",
    "bottom": "lower", "shoes": "shoes",
}


@dataclass
class ClusterConfig:
    w_emb: float = 0.6        # 임베딩 유사도 가중치 (나머지는 색)
    color_scale: float = 18.0  # 색 거리 → 유사도 변환 척도
    color_gate: float = 30.0  # 대표색 거리가 이보다 크면 같은 옷 불가
    merge: float = 0.60       # 완전 연결 병합 기준 (M0 탐색으로 보정)
    ask: float = 0.48         # 이 이상이면서 병합 안 된 묶음 쌍은 질문 후보
    max_questions: int = 30


def group_of(g: dict) -> str:
    cat = (g.get("tags") or {}).get("category")
    return CATEGORY_GROUP.get(cat, g["region"])


def similarity_matrix(garments: list[dict], emb: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """(n,n) 유사도. 같은 사진·다른 부위·색 게이트 위반은 -1."""
    n = len(garments)
    cos = emb @ emb.T if n else np.zeros((0, 0))
    S = np.full((n, n), -1.0, dtype=np.float32)
    groups = [group_of(g) for g in garments]
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = garments[i], garments[j]
            if gi["photo"] == gj["photo"] or groups[i] != groups[j]:
                continue
            # 색이 완벽히 같아도 질문 기준에 못 미치는 쌍은 색 계산을 건너뛴다
            if cfg.w_emb * float(cos[i, j]) + (1 - cfg.w_emb) < min(cfg.ask, cfg.merge):
                continue
            cd = palette_distance(gi["colors"], gj["colors"])
            if cd > cfg.color_gate:
                continue
            csim = float(np.exp(-((cd / cfg.color_scale) ** 2)))
            s = cfg.w_emb * float(cos[i, j]) + (1 - cfg.w_emb) * csim
            S[i, j] = S[j, i] = s
    return S


def cluster(garments: list[dict], S: np.ndarray, cfg: ClusterConfig,
            decisions: list[dict] | None = None) -> tuple[list[int], list[dict]]:
    """반환: (garment별 묶음 번호, 질문 큐)."""
    n = len(garments)
    idx = {g["id"]: i for i, g in enumerate(garments)}
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cannot: set[tuple[int, int]] = set()
    must: list[tuple[int, int]] = []
    for d in decisions or []:
        a, b = idx.get(d["a"]), idx.get(d["b"])
        if a is None or b is None:
            continue
        if d["verdict"] == "same":
            must.append((a, b))
        elif d["verdict"] == "different":
            cannot.add((min(a, b), max(a, b)))

    def conflict(A, B):
        return any((min(x, y), max(x, y)) in cannot or garments[x]["photo"] == garments[y]["photo"]
                   for x in A for y in B)

    # 1. 사용자가 같다고 답한 영역은 먼저 한 노드로 묶는다(확정된 한 벌)
    groups: dict[int, list[int]] = {i: [i] for i in range(n)}
    for a, b in must:
        ra, rb = find(a), find(b)
        if ra != rb and not conflict(groups[ra], groups[rb]):
            parent[rb] = ra
            groups[ra] += groups.pop(rb)

    # 2. 노드 간 유사도: 확정된 한 벌의 여러 등장 중 가장 닮은 것으로 비교(단일 연결),
    #    충돌(같은 사진·다르다는 답)이 있으면 불가
    nodes = sorted(groups)
    m = len(nodes)
    if m == n:  # 확정 묶음이 없으면 영역 유사도 그대로 (같은 사진은 이미 -1)
        T = S.copy()
        for a, b in cannot:
            T[a, b] = T[b, a] = -1.0
    else:
        T = np.full((m, m), -1.0, dtype=np.float32)
        for x in range(m):
            for y in range(x + 1, m):
                A, B = groups[nodes[x]], groups[nodes[y]]
                if conflict(A, B):
                    continue
                T[x, y] = T[y, x] = S[np.ix_(A, B)].max()

    # 3. 노드 위에서 완전 연결 병합
    owner = list(range(m))
    members = {x: {x} for x in range(m)}

    def root(x):
        while owner[x] != x:
            owner[x] = owner[owner[x]]
            x = owner[x]
        return x

    pairs = [(T[x, y], x, y) for x in range(m) for y in range(x + 1, m) if T[x, y] >= cfg.merge]
    pairs.sort(reverse=True)
    for _, x, y in pairs:
        rx, ry = root(x), root(y)
        if rx == ry:
            continue
        if T[np.ix_(sorted(members[rx]), sorted(members[ry]))].min() >= cfg.merge:
            if len(members[rx]) < len(members[ry]):
                rx, ry = ry, rx
            owner[ry] = rx
            members[rx] |= members.pop(ry)

    node_label = {}
    labels = [0] * n
    for x, nd in enumerate(nodes):
        lab = node_label.setdefault(root(x), len(node_label))
        for i in groups[nd]:
            labels[i] = lab

    by_label: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)

    def blocked(ra, rb):  # 질문 큐용: 두 묶음을 합칠 수 없는가
        return conflict(by_label[ra], by_label[rb])

    # 질문 큐: 묶음 쌍을 평균 연결 유사도로 줄 세운다. 같은 옷의 조각은 평균이 높고,
    # 비슷한 다른 옷은 한두 쌍만 높다. 대표 질문은 그 묶음 쌍에서 가장 닮은 두 영역.
    decided = {(min(idx[d["a"]], idx[d["b"]]), max(idx[d["a"]], idx[d["b"]]))
               for d in decisions or [] if d["a"] in idx and d["b"] in idx}
    cand = []
    labs = sorted(by_label)
    for x, la in enumerate(labs):
        for lb in labs[x + 1:]:
            A, B = by_label[la], by_label[lb]
            cross = S[np.ix_(A, B)]
            if cross.max() < cfg.ask:
                continue
            valid = cross[cross > -1]
            if valid.size < 0.5 * cross.size or blocked(la, lb):
                continue
            if any((min(i, j), max(i, j)) in decided for i in A for j in B):
                continue
            i, j = np.unravel_index(int(cross.argmax()), cross.shape)
            cand.append((float(valid.mean()), A[i], B[j]))
    cand.sort(reverse=True)
    questions = [{"a": garments[i]["id"], "b": garments[j]["id"], "score": round(s, 3)}
                 for s, i, j in cand[: cfg.max_questions]]
    return labels, questions
