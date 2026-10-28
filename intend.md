# intend — 사용자가 할 일

작성 2026-10-28 · 지금 단계: **M0 (사진 50장으로 갤러리 스캐너 검증)**

지금 스캐너는 합성 사진으로만 확인했습니다. 실제 사진에서 얼마나 맞는지는 윤경 님 사진으로만 알 수 있습니다.
아래 1~5를 하면 M0가 끝나고, 결과에 따라 앱(M1) 착수와 모델을 정합니다. 총 1시간 30분쯤 걸립니다.

| 순서 | 할 일 | 시간 | 결과물 |
| --- | --- | --- | --- |
| 1 | PR #1 합치기, Mac에 준비 | 10분 | `~/closet-ai` |
| 2 | 사진 50장 고르기 | 15분 | `m0/photos/` |
| 3 | 정답 라벨 달기 (스캔 전에) | 45분 | `m0/labels.csv` |
| 4 | 스캔과 평가 실행 | 5분 | `out/m0-result.md`, `out/report.html` |
| 5 | 결과를 스레드에 올리기 | 1분 | 다음 결정 |

## 1. 준비 (한 번만)

1. [PR #1](https://github.com/cogito5170/closet-ai/pull/1)에서 **Merge pull request**를 누릅니다.
2. 터미널에서 실행합니다. `uv`가 이미 있으면 첫 줄은 건너뜁니다.

```
brew install uv
cd ~
git clone https://github.com/cogito5170/closet-ai.git
cd ~/closet-ai/scanner
uv sync
uv run closet-scan models
```

마지막 줄이 `모델 위치: ...`를 출력하면 준비가 끝난 것입니다. Apple Silicon Mac에서 동작하고, Intel Mac은 지원하지 않습니다.

## 2. 사진 50장 고르기

`~/closet-ai/m0/photos/` 폴더를 만들어 사진을 넣습니다. 이 폴더는 GitHub에 올라가지 않습니다. iPhone HEIC 파일을 그대로 넣어도 됩니다.

```
mkdir -p ~/closet-ai/m0/photos
open ~/closet-ai/m0/photos
```

- 구성: 거울 셀카 20장, 전신 OOTD 20장, 상반신 셀카 10장
- **같은 옷이 여러 사진에 반복해서 나오게** 고릅니다. 반복이 없으면 같은 옷 판별을 잴 수 없습니다.
- 비슷하지만 다른 옷(예: 검정 상의 두 벌)이 나오는 사진을 일부러 넣습니다.
- 넣지 않을 사진: 여러 명이 나온 사진, 사람 없이 옷만 찍은 사진(스캐너가 아직 다루지 않습니다)

## 3. 정답 라벨 달기 (스캔보다 먼저)

스캐너 결과를 먼저 보면 그 답에 끌려 정답이 오염됩니다(기준점 효과). 반드시 라벨을 먼저 답니다.

```
open -a Safari ~/closet-ai/m0/label.html
```

1. **사진 폴더 열기**를 누르고 `m0/photos`를 고릅니다.
2. 사진마다 보이는 옷을 모두 등록합니다. 대상은 상의·하의·아우터·원피스·신발이고, 가방·모자·액세서리는 적지 않아도 됩니다.
   - 처음 나온 옷은 **새 옷 추가**에서 이름, 종류, 색을 넣습니다.
   - 이미 등록한 옷은 **지금까지 등록한 옷에서 고르기**에서 누릅니다. 같은 옷은 모든 사진에서 같은 번호여야 합니다.
   - 겉옷 안에 상의가 보이면 둘 다 등록합니다.
3. 한 장이 끝나면 `Enter`로 완료 표시를 하고 다음 사진으로 넘어갑니다. `←` `→`로 이동할 수 있습니다.
4. 다 끝나면 **labels.csv 내려받기**를 누르고 파일을 옮깁니다.

```
mv ~/Downloads/labels.csv ~/closet-ai/m0/labels.csv
```

진행 상황은 브라우저에 자동 저장됩니다. 중간에 닫아도 같은 폴더를 다시 열면 이어집니다.

## 4. 스캔과 평가

```
cd ~/closet-ai/scanner
uv run closet-scan scan ../m0/photos --out ../out
uv run closet-scan eval ../out ../m0/labels.csv --sweep --md ../out/m0-result.md
open ../out/report.html
```

- `m0-result.md`: 지표별 값과 합격·불합격 표입니다. 기준은 미리 정해 두었습니다(검출 ≥ 85%, 같은 옷 판별 F1 ≥ 0.80, 과병합 ≤ 5%, 옷 10벌당 수정 ≤ 3).
- `report.html`: 찾은 옷과 "같은 옷인가요?" 질문이 있습니다. 평가가 끝난 뒤에 답해 보세요. 답하는 경험 자체가 앱의 확인 화면 시안입니다.

## 5. 결과 올리기

이 스레드에 `~/closet-ai/out/m0-result.md`를 올리고, `report.html`을 보며 느낀 점을 한두 줄 적어 주세요. 예를 들면 "이 옷을 못 찾았다", "질문이 너무 많다" 같은 것입니다.
`out/` 폴더에는 옷 crop 이미지가 있어 GitHub에 올라가지 않게 해 두었습니다.

## 선택: Claude 태깅까지 비교하기

원피스·아우터 같은 세부 분류와 옷 이름은 Claude 태깅을 켜야 나옵니다. Anthropic API 키가 있으면 아래를 실행하고, 결과 파일도 함께 올려 주세요. 사진 50장에 드는 비용은 1달러를 크게 밑돕니다(Claude Haiku 5.5, 옷 crop 1장당 1회 호출). 이 값은 추정이고, 실행 끝에 실제 토큰 수가 출력됩니다.

```
cd ~/closet-ai/scanner
uv sync --extra claude
export ANTHROPIC_API_KEY=여기에_키
uv run closet-scan scan ../m0/photos --out ../out-claude --tagger claude --verify
uv run closet-scan eval ../out-claude ../m0/labels.csv --md ../out-claude/m0-result.md
```

## M0 이후에 정할 것 (지금은 하지 않아도 됩니다)

- 스캐너 모델 확정: M0 결과를 보고 Claude가 추천안을 드립니다(PLAN §15).
- 앱 이름
- 스타일 유형 분류(8~16개)와 테스트 문항
